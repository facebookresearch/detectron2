# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
)


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseDistributionLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U coordinate distribution, tensor of shape [N, Cb * C, S, S]
     * V coordinate distribution, tensor of shape [N, Cb * C, S, S]
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
        self.w_dis        = cfg.MODEL.BLOCK.DISTRI_WEIGHTS
        self.smoothing    = 0.1
        self.confidence   = 1. - self.smoothing
        self.l1_loss      = cfg.MODEL.BLOCK.L1_LOSS

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

        accumulator = ChartBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if packed_annotations is None:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        h, w = densepose_predictor_outputs.fine_segm.shape[2:]
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

        return {**losses_uv, **losses_segm}

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
        loss = {
            "loss_densepose_dis_u": densepose_predictor_outputs.u_dis.sum() * 0,
            "loss_densepose_dis_v": densepose_predictor_outputs.v_dis.sum() * 0
        }
        if self.l1_loss:
            loss.update({
                "loss_densepose_U": densepose_predictor_outputs.u_dis.sum() * 0,
                "loss_densepose_V": densepose_predictor_outputs.v_dis.sum() * 0,
            })
        return

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

        u_gt = packed_annotations.u_gt[j_valid_fg]
        v_gt = packed_annotations.v_gt[j_valid_fg]
        block_width = 1. / self.block_num
        u_left = (u_gt / block_width).int()
        u_right = u_left + 1
        u_right[u_right > self.block_num] = 0
        u_weight_left = (u_right * block_width) - u_gt
        u_weight_right = u_gt - (u_left * block_width)
        v_left = (v_gt / block_width).int()
        v_right = v_left + 1
        v_right[v_right > self.block_num] = 0
        v_weight_left = (v_right * block_width) - v_gt
        v_weight_right = v_gt - (v_left * block_width)

        est_shape = densepose_predictor_outputs.u_dis.shape

        u_est_dis = interpolator.extract_at_points(
            densepose_predictor_outputs.u_dis.reshape(est_shape[0], -1, self.block_num + 1, est_shape[2], est_shape[3]),
            block=True,
            block_slice=slice(None),
            slice_fine_segm=fine_segm_gt,
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[j_valid_fg, :]
        v_est_dis = interpolator.extract_at_points(
            densepose_predictor_outputs.v_dis.reshape(est_shape[0], -1, self.block_num + 1, est_shape[2], est_shape[3]),
            block=True,
            block_slice=slice(None),
            slice_fine_segm=fine_segm_gt,
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[j_valid_fg, :]

        # if u_right.max() > 10 or v_right.max() > 10:
        #     print("u_left [{},{}] u_right [{},{}] v_left [{},{}] v_right[{},{}]".format(u_left.min(), u_left.max(), u_right.min()
        #                                                                         , u_right.max(), v_left.min(), v_left.max(),
        #                                                                         v_right.min(), v_right.max()))

        loss_dis_u = F.cross_entropy(u_est_dis, u_left.long(), reduction="none") * u_weight_left + F.cross_entropy(
            u_est_dis, u_right.long(), reduction="none") * u_weight_right
        loss_dis_v = F.cross_entropy(v_est_dis, v_left.long(), reduction="none") * v_weight_left + F.cross_entropy(
            v_est_dis, v_right.long(), reduction="none") * v_weight_right

        loss = {
            "loss_densepose_dis_u": loss_dis_u.mean() * self.w_dis,
            "loss_densepose_dis_v": loss_dis_v.mean() * self.w_dis
        }

        if self.l1_loss:
            block_value = torch.linspace(0, self.block_num, self.block_num + 1)
            block_value *= 1. / self.block_num
            block_value = block_value.to(u_est_dis[0].device)
            u_est = F.linear(F.softmax(u_est_dis, dim=1), block_value.float())
            v_est = F.linear(F.softmax(v_est_dis, dim=1), block_value.float())
            loss.update({
                "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_dis,
                "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_dis
            })

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
