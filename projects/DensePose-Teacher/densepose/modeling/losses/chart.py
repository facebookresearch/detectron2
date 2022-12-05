# Copyright (c) Facebook, Inc. and its affiliates.

from struct import pack
from typing import Any, List, Dict
import torch
from torch.nn import functional as F
import numpy as np
import math

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from densepose.modeling.correction import CorrectorPredictorOutput
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    resample_data,
)


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U coordinates, tensor of shape [N, C, S, S]
     * V coordinates, tensor of shape [N, C, S, S]
     * fine segmentation estimates, tensor of shape [N, C, S, S] with raw unnormalized
       scores for each fine segmentation label at each location
     * coarse segmentation estimates, tensor of shape [N, D, S, S] with raw unnormalized
       scores for each coarse segmentation label at each location
    where N is the number of detections, C is the number of fine segmentation
    labels, S is the estimate size ( = width = height) and D is the number of
    coarse segmentation channels.

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
        # self.segm_loss = MaskOrSegmentationLoss(cfg)

        # self.w_pseudo     = cfg.MODEL.SEMI.UNSUP_WEIGHTS
        self.w_p_segm     = cfg.MODEL.SEMI.SEGM_WEIGHTS
        self.w_p_points   = cfg.MODEL.SEMI.POINTS_WEIGHTS
        self.pseudo_threshold = cfg.MODEL.SEMI.THRESHOLD
        self.n_channels = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.w_crt_segm = cfg.MODEL.SEMI.COR.SEGM_WEIGHTS
        self.w_crt_sigma = cfg.MODEL.SEMI.COR.SIGMA_WEIGHTS

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, **kwargs
    ) -> LossDict:

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        accumulator = ChartBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        if packed_annotations is None:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        h, w = densepose_predictor_outputs.u.shape[2:]
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
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        losses = {
            "loss_densepose_U": densepose_predictor_outputs.u.sum() * 0,
            "loss_densepose_V": densepose_predictor_outputs.v.sum() * 0,
        }
        if self.uv_confidence:
            losses.update({
                "loss_correction_UV": densepose_predictor_outputs.crt_sigma.sum() * 0
            })
        return losses

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        losses = {
            "loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_densepose_S": densepose_predictor_outputs.coarse_segm.sum() * 0,
        }
        losses.update({
            "loss_correction_I": densepose_predictor_outputs.crt_segm.sum() * 0,
            "loss_correction_S": densepose_predictor_outputs.crt_segm.sum() * 0,
        })

        return losses

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        u_gt = packed_annotations.u_gt[j_valid_fg]
        u_est = interpolator.extract_at_points(densepose_predictor_outputs.u)[j_valid_fg]
        v_gt = packed_annotations.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points(densepose_predictor_outputs.v)[j_valid_fg]
        loss_u = F.smooth_l1_loss(u_est, u_gt, reduction="none")
        loss_v = F.smooth_l1_loss(v_est, v_gt, reduction="none")

        # sigma = interpolator.extract_at_points(densepose_predictor_outputs.crt_sigma)[j_valid_fg]
        sigma = interpolator.extract_at_points(densepose_predictor_outputs.crt_sigma)[interpolator.j_valid]
        delta_t_delta = (u_est - u_gt.detach()) ** 2 + (v_est - v_gt.detach()) ** 2
        # sigma = F.softplus(sigma) + 1e-9
        # uv_weights = torch.ones_like(loss_u, dtype=torch.float32)
        loss = {
            # "loss_correction_UV": (self.log2pi + torch.log(sigma).sum(dim=1) + delta_t_delta).sum() * 0.5 *
            #                       self.w_crt_sigma
            "loss_correction_UV": F.mse_loss(sigma, torch.sqrt(delta_t_delta), reduction='sum') * self.w_crt_sigma
        }

        loss.update({
            "loss_densepose_U": (loss_u).sum() * self.w_points,
            "loss_densepose_V": (loss_v).sum() * self.w_points,
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

        if packed_annotations.coarse_segm_gt is None:
            loss_coarse_segm = densepose_predictor_outputs.coarse_segm.sum() * 0
        coarse_segm_est = densepose_predictor_outputs.coarse_segm[packed_annotations.bbox_indices]
        with torch.no_grad():
            coarse_segm_gt = resample_data(
                packed_annotations.coarse_segm_gt.unsqueeze(1),
                packed_annotations.bbox_xywh_gt,
                packed_annotations.bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)
        if self.n_segm_chan == 2:
            loss_coarse_segm = F.cross_entropy(coarse_segm_est, (coarse_segm_gt > 0).long())
        else:
            loss_coarse_segm = F.cross_entropy(coarse_segm_est, coarse_segm_gt.long())

        loss = {
            "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part,
            "loss_densepose_S": loss_coarse_segm * self.w_segm,
        }

        fine_segm_crt_est, coarse_segm_crt_est = densepose_predictor_outputs.crt_segm[:, :self.n_channels], densepose_predictor_outputs.crt_segm[:, -1]
        fine_segm_crt_est = interpolator.extract_at_points(fine_segm_crt_est)[interpolator.j_valid]
        coarse_segm_crt_est = coarse_segm_crt_est[packed_annotations.bbox_indices]
        segm_est_index = fine_segm_est.argmax(dim=1).long()
        fine_segm_crt_gt = fine_segm_gt.detach() == segm_est_index

        crt_fine_segm_loss = F.binary_cross_entropy_with_logits(fine_segm_crt_est, fine_segm_crt_gt.float(), reduction='none')
        # fine_weights = fine_segm_crt_gt.sum() / (~fine_segm_crt_gt).sum()
        # crt_fine_segm_loss[~fine_segm_crt_gt] *= fine_weights

        segm_est_index = coarse_segm_est.argmax(dim=1).long()
        coarse_segm_crt_gt = (coarse_segm_gt.detach() > 0) == segm_est_index
        # coarse_segm_crt_gt = (coarse_segm_gt.detach() > 0).float()
        # focal_weights = torch.sigmoid(coarse_segm_crt_est.detach())
        # focal_weights[coarse_segm_crt_gt] = 1 - focal_weights[coarse_segm_crt_gt]
        coarse_segm_loss = F.binary_cross_entropy_with_logits(coarse_segm_crt_est, coarse_segm_crt_gt.float(), reduction='none')
        # coarse_weights = coarse_segm_crt_gt.sum() / (~coarse_segm_crt_gt).sum()
        # coarse_segm_loss[~coarse_segm_crt_gt] *= coarse_weights

        loss.update({
            "loss_correction_I": crt_fine_segm_loss.mean() * self.w_crt_segm,
            "loss_correction_S": coarse_segm_loss.mean() * self.w_crt_segm
        })

        return loss
