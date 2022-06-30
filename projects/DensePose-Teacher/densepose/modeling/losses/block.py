# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    BlockBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    resample_data
)


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseBlockLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U classification, tensor of shape [N, Cb, S, S]
     * U offset, tensor of shape [N, Cb, S, S]
     * V classification, tensor of shape [N, Cb, S, S]
     * V offset, tensor of shape [N, Cb, S, S]
     * fine segmentation estimates, tensor of shape [N, C, S, S] with raw unnormalized
       scores for each fine segmentation label at each location
     * coarse segmentation estimates, tensor of shape [N, D, S, S] with raw unnormalized
       scores for each coarse segmentation label at each location
    where N is the number of detections, C is the number of fine segmentation
    labels, S is the estimate size ( = width = height) and D is the number of
    coarse segmentation channels. Cb is the number of blocks.

    The losses are:
    * regression (smooth L1) loss for U and V coordinates
    * cross entropy loss for fine (I) and coarse (S) segmentations
    Each loss has an associated weight
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize chart-based loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.segm_loss = MaskOrSegmentationLoss(cfg)

        self.n_i_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES
        self.gamma = 0.99
        self.block_num    = cfg.MODEL.BLOCK.BLOCK_NUM
        self.w_block_cls  = cfg.MODEL.BLOCK.CLS_WEIGHTS
        self.w_block_reg  = cfg.MODEL.BLOCK.REGRESS_WEIGHTS
        self.smoothing    = 0.1
        self.confidence   = 1. - self.smoothing

        self.w_pseudo = cfg.MODEL.SEMI.UNSUP_WEIGHTS
        self.w_p_segm = cfg.MODEL.SEMI.SEGM_WEIGHTS
        self.w_p_points = cfg.MODEL.SEMI.POINTS_WEIGHTS
        self.pseudo_threshold = cfg.MODEL.SEMI.THRESHOLD
        self.n_channels = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.loss_name = cfg.MODEL.SEMI.LOSS_NAME

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, **kwargs
    ) -> LossDict:
        """
        Produce chart-based DensePose losses

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
                * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
                * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
                * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
            where N is the number of detections, C is the number of fine segmentation
            labels, S is the estimate size ( = width = height) and D is the number of
            coarse segmentation channels.

        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels;
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
        """
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        accumulator = BlockBasedAnnotationsAccumulator(self.block_num)
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if packed_annotations is None:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        h, w = densepose_predictor_outputs.u_cls.shape[2:]
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations,
            (h, w),
        )

        j_valid_fg = interpolator.j_valid * (  # pyre-ignore[16]
            packed_annotations.fine_segm_labels_gt > 0
        )
        if not torch.any(j_valid_fg):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        losses_uv = self.produce_densepose_losses_uv(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )

        losses_segm = self.produce_densepose_losses_segm(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )

        losses_unsup = self.produce_densepose_losses_unsup(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
        )

        return {**losses_uv, **losses_segm, **losses_unsup}

    def produce_fake_densepose_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine segmentation and U/V coordinates. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0
        """
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for U/V coordinates. These are used when no suitable ground
        truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
        """
        return {
            "loss_densepose_U_cls": densepose_predictor_outputs.u_cls.sum() * 0,
            "loss_densepose_U_offset": densepose_predictor_outputs.u_offset.sum() * 0,
            "loss_densepose_V_cls": densepose_predictor_outputs.v_cls.sum() * 0,
            "loss_densepose_V_offset": densepose_predictor_outputs.v_offset.sum() * 0,
        }

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine / coarse segmentation. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
        """
        losses = {
            "loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_densepose_S": self.segm_loss.fake_value(densepose_predictor_outputs),
        }
        return losses

    def produce_fake_densepose_losses_unsup(self, densepose_predictor_outputs: Any) -> LossDict:
        return {
            "loss_unsup_segm": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_unsup_u": densepose_predictor_outputs.u.sum() * 0,
            "loss_unsup_v": densepose_predictor_outputs.v.sum() * 0,
        }

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Compute losses for U/V coordinates: smooth L1 loss between
        estimated coordinates and the ground truth.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u_cls - U blocks classification estimates per fine labels, tensor of shape [N, C * Cb, S, S]
             * u_offset - U block offset estimates per fine labels, tensor of shape [N, C * Cb, S, S]
             * v_cls - V blocks classification estimates per fine labels, tensor of shape [N, C * Cb, S, S]
             * v_offset - V block offset estimates per fine labels, tensor of shape [N, C * Cb, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U_cls`: LbelSmoothing negative log loss for U coordinate estimates
             * `loss_densepose_U_offset`: smooth L1 loss for U offset
             * `loss_densepose_V_cls`: LbelSmoothing negative log loss for V coordinate estimates
             * `loss_densepose_V_offset`: smooth L1 loss for V offset
        """
        fine_segm_gt = packed_annotations.fine_segm_labels_gt - 1

        u_gt = packed_annotations.u_gt
        v_gt = packed_annotations.v_gt
        u_gt_cls = packed_annotations.u_gt_cls
        u_gt_offsets = packed_annotations.u_gt_offsets
        v_gt_cls = packed_annotations.v_gt_cls
        v_gt_offsets = packed_annotations.v_gt_offsets

        est_shape = densepose_predictor_outputs.u_cls.shape
        u_est_cls = interpolator.extract_at_points(
            densepose_predictor_outputs.u_cls.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block=True,
            block_slice=slice(None),
            slice_fine_segm=fine_segm_gt,
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[j_valid_fg, :]
        u_est_offsets = interpolator.extract_at_points(
            densepose_predictor_outputs.u_offset.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block=True,
            block_slice=u_gt_cls,
            slice_fine_segm=fine_segm_gt
        )[j_valid_fg]
        v_est_cls = interpolator.extract_at_points(
            densepose_predictor_outputs.v_cls.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block=True,
            block_slice=slice(None),
            slice_fine_segm=fine_segm_gt,
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[j_valid_fg, :]
        v_est_offsets = interpolator.extract_at_points(
            densepose_predictor_outputs.v_offset.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block=True,
            block_slice=v_gt_cls,
            slice_fine_segm=fine_segm_gt
        )[j_valid_fg]

        u_gt_cls = u_gt_cls[j_valid_fg]
        v_gt_cls = v_gt_cls[j_valid_fg]
        u_gt_offsets = u_gt_offsets[j_valid_fg]
        v_gt_offsets = v_gt_offsets[j_valid_fg]

        return {
            "loss_densepose_U_cls": torch.sum(self.label_smoothing(u_est_cls, u_gt_cls.long()))*self.w_block_cls,
            "loss_densepose_U_offset": F.smooth_l1_loss(u_est_offsets, u_gt_offsets, reduction="sum")*self.w_block_reg,
            "loss_densepose_V_cls": torch.sum(self.label_smoothing(v_est_cls, v_gt_cls.long()))*self.w_block_cls,
            "loss_densepose_V_offset": F.smooth_l1_loss(v_est_offsets, v_gt_offsets, reduction="sum")*self.w_block_reg,
        }

    def label_smoothing(self, x, target, bce=False):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = F.nll_loss(logprobs, target)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = nll_loss * self.confidence + smooth_loss * self.smoothing
        return loss

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Losses for fine / coarse segmentation: cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points for fine segmentation and dense mask annotations
        for coarse segmentation.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """
        fine_segm_gt = packed_annotations.fine_segm_labels_gt[
            interpolator.j_valid  # pyre-ignore[16]
        ]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
        )[interpolator.j_valid, :]
        return {
            "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part,
            "loss_densepose_S": self.segm_loss(
                proposals_with_gt, densepose_predictor_outputs, packed_annotations
            )
            * self.w_segm,
        }


    def produce_densepose_losses_unsup(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
    ) -> LossDict:
        losses = {}
        pseudo_keys = ["fine_segm_p", "u_p", "v_p"]
        est_keys = ["fine_segm", "u_cls", "v_cls"]
        mask = None
        index = None
        weights = None
        for p_key, e_key in zip(pseudo_keys, est_keys):
            if getattr(packed_annotations, p_key) is None:
                return self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)
            est = getattr(densepose_predictor_outputs, e_key)[packed_annotations.bbox_indices]
            with torch.no_grad():
                pseudo = getattr(packed_annotations, p_key)
                pseudo = resample_data(
                    pseudo,
                    packed_annotations.bbox_xywh_gt,
                    packed_annotations.bbox_xywh_est,
                    self.heatmap_size,
                    self.heatmap_size,
                    mode="nearest",
                    padding_mode="zeros",
                )
            if p_key == "fine_segm_p":
                est = est.permute(0, 2, 3, 1).reshape(-1, self.n_channels)
                pseudo = pseudo.permute(0, 2, 3, 1).reshape(-1, self.n_channels)
                mask, index = torch.max(F.softmax(pseudo, dim=1), dim=1)
                if self.pseudo_threshold < 1:
                    mask = mask >= self.pseudo_threshold
                    if mask.sum() <= 0:
                        return self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)
                    index = index[mask].long()
                    est = est[mask]
                loss = F.cross_entropy(est, index.long(), reduction='none')
                if self.loss_name == "sce":
                    label_one_hot = F.one_hot(torch.clamp(index, min=0, max=self.n_channels - 1),
                                              self.n_channels).float()
                    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
                    rce_loss = -1 * est * torch.log(label_one_hot)
                    rce_loss = torch.sum(rce_loss, dim=1)
                    rce_loss = torch.mean(rce_loss * mask)
                    loss = loss * (1 - mask) + rce_loss
                    weights = mask / mask.sum()
                    losses.update({"loss_unsup_segm": (loss * weights).sum() * self.w_pseudo * self.w_p_segm})
                elif self.loss_name == "ce":
                    losses.update({"loss_unsup_segm": loss.mean() * self.w_pseudo * self.w_p_segm})
                # losses.update({"loss_unsup_segm": F.cross_entropy(est, torch.argmax(pseudo, dim=1).long()) * self.w_pseudo * self.w_p_segm})
            else:
                est = est.permute(0, 2, 3, 1).reshape(-1, self.n_channels - 1, self.block_num)
                pseudo = pseudo.permute(0, 2, 3, 1).reshape(-1, self.n_channels - 1, self.block_num)
                if self.pseudo_threshold < 1:
                    pseudo = pseudo[mask]
                    est = est[mask]
                # 0 is background
                index = index[index != 0] - 1
                est = est[np.arange(index.shape[0]), index]
                pseudo = pseudo[np.arange(index.shape[0]), index]
                block_index = torch.argmax(F.softmax(pseudo, dim=1), dim=1)
                loss = F.cross_entropy(est, block_index.long(), reduction='none')
                if self.loss_name == "sce":
                    label_one_hot = F.one_hot(torch.clamp(block_index, min=0, max=self.n_channels - 1),
                                              self.n_channels).float()
                    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
                    rce_loss = -1 * est * torch.log(label_one_hot)
                    rce_loss = torch.sum(rce_loss, dim=1)
                    rce_loss = torch.mean(rce_loss * mask)
                    loss = loss * (1 - mask) + rce_loss
                    weights = mask / mask.sum()
                    losses.update({"loss_unsup_segm": (loss * weights).sum() * self.w_pseudo * self.w_p_points})
                elif self.loss_name == "ce":
                    losses.update({"loss_{}".format(p_key): loss.mean() * self.w_pseudo * self.w_p_points})
                # losses.update({"loss_{}".format(p_key): F.smooth_l1_loss(est, pseudo) * self.w_pseudo * self.w_p_points})
        return losses
