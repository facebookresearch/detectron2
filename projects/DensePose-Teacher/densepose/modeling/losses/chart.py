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

        # self.w_crt_points = cfg.MODEL.SEMI.COR.POINTS_WEIGHTS
        self.w_crt_segm = cfg.MODEL.SEMI.COR.SEGM_WEIGHTS

        self.total_iteration = cfg.SOLVER.MAX_ITER
        self.warm_up_iter = cfg.MODEL.SEMI.COR.WARM_ITER

        self.uv_confidence = cfg.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.ENABLED
        self.log2pi = math.log(2 * math.pi)
        self.w_crt_sigma = cfg.MODEL.SEMI.COR.SIGMA_WEIGHTS
        self.w_p_segm_scale = cfg.MODEL.SEMI.SEGM_SCALE
        self.ts = cfg.MODEL.SEMI.COR.TS

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, iteration, dp_predictions, **kwargs
    ) -> LossDict:

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        assert iteration != -1, ("iteration should not be -1!")

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

        losses_segm, front_index = self.produce_densepose_losses_segm(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
            dp_predictions=dp_predictions
        )

        losses_unsup = self.produce_densepose_losses_unsup(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            iteration = iteration,
            front_index = front_index,
        )

        return {**losses_uv, **losses_segm, **losses_unsup}

    def produce_fake_densepose_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        losses_unsup = self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm, **losses_unsup}

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

    def produce_fake_densepose_losses_unsup(self, densepose_predictor_outputs: Any) -> LossDict:
        return {
            "loss_unsup_segm": densepose_predictor_outputs.fine_segm.sum() * 0,
            # "loss_unsup_u": densepose_predictor_outputs.u.sum() * 0,
            # "loss_unsup_v": densepose_predictor_outputs.v.sum() * 0,
        }

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        losses = {
            "loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_densepose_S": densepose_predictor_outputs.coarse_segm.sum() * 0,
        }
        if not self.ts:
            losses.update({
                "loss_correction_IS": densepose_predictor_outputs.crt_segm.sum() * 0,
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

        if self.uv_confidence:
            sigma = interpolator.extract_at_points(densepose_predictor_outputs.crt_sigma)[j_valid_fg]
            sigma = F.softplus(sigma) + 0.01
            delta_t_delta = (u_est.detach() - u_gt.detach()) ** 2 + (v_est.detach() - v_gt.detach()) ** 2
            uv_weights = (1 / sigma.detach()).clamp(0., 2.)
            loss = {
                "loss_correction_UV": (self.log2pi + 2 * torch.log(sigma) + delta_t_delta / sigma).sum() * 0.5 *
                                      self.w_crt_sigma
            }
        else:
            loss = {}
            uv_weights = torch.ones_like(loss_u, dtype=torch.float32)

        loss.update({
            "loss_densepose_U": (loss_u * uv_weights).sum() * self.w_points,
            "loss_densepose_V": (loss_v * uv_weights).sum() * self.w_points,
        })

        return loss

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
        dp_predictions: Dict,
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

        if not self.ts:
            fine_segm_crt_est = interpolator.extract_at_points(
                densepose_predictor_outputs.crt_segm,
                slice_fine_segm=slice(None),
                w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
            )[interpolator.j_valid, :].squeeze(1)
            # coarse_segm_crt_est = densepose_predictor_outputs.crt_segm[packed_annotations.bbox_indices]
            # coarse_segm_crt_est = coarse_segm_crt_est[:, 1]
            segm_est_index = fine_segm_est.detach().argmax(dim=1).long()
            fine_segm_crt_gt = fine_segm_gt.detach() == segm_est_index

            one_loss = fine_segm_crt_gt.sum().detach()
            zero_loss = (~fine_segm_crt_gt).sum().detach()

            crt_fine_segm_loss = F.binary_cross_entropy_with_logits(fine_segm_crt_est, fine_segm_crt_gt.float(), reduction='none', pos_weight=zero_loss / one_loss)
            # crt_fine_segm_loss[~fine_segm_crt_gt] *= (one_loss / zero_loss)

            # segm_est_index = coarse_segm_est.detach().argmax(dim=1).long()
            # coarse_segm_crt_gt = (coarse_segm_gt.detach() > 0) == segm_est_index

            # coarse_one_loss = coarse_segm_crt_gt.sum().detach()
            # coarse_zero_loss = (~coarse_segm_crt_gt).sum().detach()
            # coarse_segm_loss = F.binary_cross_entropy_with_logits(coarse_segm_crt_est, coarse_segm_crt_gt.float(), reduction='none')
            # coarse_segm_loss[~coarse_segm_crt_gt] *= (coarse_one_loss / coarse_zero_loss)

            # if len(dp_predictions['pred']) > 0:
            #     dp_crt_est = interpolator.extract_at_points(
            #         torch.cat(dp_predictions['pred'], dim=1),
            #         slice_fine_segm=slice(None),
            #         w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            #         w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            #         w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            #         w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
            #     )[interpolator.j_valid, :].reshape[-1,]

            loss.update({
                "loss_correction_IS": (crt_fine_segm_loss.mean() * self.w_crt_segm)
                                       # + coarse_segm_loss.mean())
                                       # * 0.5 * self.w_crt_segm
            })

        return loss, (coarse_segm_gt > 0).reshape(-1, )


    def produce_densepose_losses_unsup(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        iteration,
        front_index,
    ) -> LossDict:
        if getattr(packed_annotations, "pseudo_segm") is None:
            return self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)

        if 0 <= iteration < self.warm_up_iter:
            factor = np.exp(-5 * (1 - iteration / self.warm_up_iter) ** 2)
        else:
            factor = 1.

        est = getattr(densepose_predictor_outputs, "fine_segm")[packed_annotations.bbox_indices]
        est = est.permute(0, 2, 3, 1).reshape(-1, self.n_channels)
        with torch.no_grad():
            # pos_index = getattr(packed_annotations, "pseudo_mask")
            # pos_index = resample_data(
            #     pos_index,
            #     packed_annotations.bbox_xywh_gt,
            #     packed_annotations.bbox_xywh_est,
            #     self.heatmap_size,
            #     self.heatmap_size,
            #     mode="nearest",
            #     padding_mode="zeros",
            # )
            # pos_index = torch.sigmoid(pos_index).permute(0, 2, 3, 1).reshape(-1, ) > 0.5
            # pos_index = pos_index[:, 0]
            # pos_index = pos_index[:, 0] * pos_index[:, 1]
            pseudo_segm = getattr(packed_annotations, "pseudo_segm")

            h, w = pseudo_segm.shape[2:]
            interpolator = BilinearInterpolationHelper.from_matches_to_pseudo(
                packed_annotations, (h, w)
            )
            if not torch.any(interpolator.j_valid):
                return self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)

            sampled_pseudo_segm = interpolator.extract_at_points_pseduo(
                pseudo_segm,
                slice_fine_segm=slice(None),
                w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
            )[interpolator.j_valid]
            sampled_gt = packed_annotations.fine_segm_labels_gt[
                interpolator.j_valid
            ]
            neg_index = sampled_pseudo_segm.argmax(dim=1) != sampled_gt
            entropy = sampled_pseudo_segm[neg_index]
            entropy = torch.sum(F.softmax(entropy, dim=1) * -1 * F.log_softmax(entropy, dim=1), dim=1)
            mu = entropy.mean()
            thres = (entropy - mu).abs().mean()
            thres = mu - 3 * thres

            pseudo_segm = resample_data(
                pseudo_segm,
                packed_annotations.bbox_xywh_gt,
                packed_annotations.bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).permute(0, 2, 3, 1).reshape(-1, self.n_channels)
            pseudo_entropy = torch.sum(F.softmax(pseudo_segm, dim=1) * -1 * F.log_softmax(pseudo_segm, dim=1), dim=1)
            pos_index = pseudo_entropy < thres
            pos_index = pos_index * front_index

            if pos_index.sum() <= 0:
                return self.produce_fake_densepose_losses_unsup(densepose_predictor_outputs)

            pred_index = pseudo_segm[pos_index].argmax(dim=1)

        loss = F.cross_entropy(est[pos_index], pred_index.long(), reduction='mean')
        losses = {"loss_unsup_segm": loss * self.w_p_segm * factor}

        # u_est = getattr(densepose_predictor_outputs, "u")[packed_annotations.bbox_indices]
        # v_est = getattr(densepose_predictor_outputs, "v")[packed_annotations.bbox_indices]
        # u_est = (u_est.permute(0, 2, 3, 1).reshape(-1, self.n_channels)[pos_index])
        # v_est = (v_est.permute(0, 2, 3, 1).reshape(-1, self.n_channels)[pos_index])
        # u_est = u_est[np.arange(u_est.shape[0]), pred_index]
        # v_est = v_est[np.arange(v_est.shape[0]), pred_index]
        #
        # with torch.no_grad():
        #     pseudo_u = getattr(packed_annotations, "pseudo_u")
        #     pseudo_v = getattr(packed_annotations, "pseudo_v")
        #     pseudo_u = resample_data(
        #         pseudo_u,
        #         packed_annotations.bbox_xywh_gt,
        #         packed_annotations.bbox_xywh_est,
        #         self.heatmap_size,
        #         self.heatmap_size,
        #         mode="nearest",
        #         padding_mode="zeros",
        #     ).permute(0, 2, 3, 1).reshape(-1, self.n_channels)[pos_index]
        #
        #     pseudo_v = resample_data(
        #         pseudo_v,
        #         packed_annotations.bbox_xywh_gt,
        #         packed_annotations.bbox_xywh_est,
        #         self.heatmap_size,
        #         self.heatmap_size,
        #         mode="nearest",
        #         padding_mode="zeros",
        #     ).permute(0, 2, 3, 1).reshape(-1, self.n_channels)[pos_index]
        #
        #     pseudo_u = pseudo_u[np.arange(pseudo_u.shape[0]), pred_index]#.clamp(0., 1.)
        #     pseudo_v = pseudo_v[np.arange(pseudo_v.shape[0]), pred_index]#.clamp(0., 1.)
        #
        #     if self.uv_confidence:
        #         with torch.no_grad():
        #             pseudo_sigma = getattr(packed_annotations, "pseudo_sigma")
        #             pseudo_sigma = resample_data(
        #                 pseudo_sigma,
        #                 packed_annotations.bbox_xywh_gt,
        #                 packed_annotations.bbox_xywh_est,
        #                 self.heatmap_size,
        #                 self.heatmap_size,
        #                 mode="nearest",
        #                 padding_mode="zeros",
        #             ).permute(0, 2, 3, 1).reshape(-1, self.n_channels)[pos_index]
        #             pseudo_sigma = pseudo_sigma[np.arange(pseudo_sigma.shape[0]), pred_index]
        #             pseudo_sigma = 0.5 / (F.softplus(pseudo_sigma) + 0.01)
        #             pseudo_sigma = pseudo_sigma.clamp(0., 1.)
        #     else:
        #         pseudo_sigma = torch.ones_like(pseudo_u, dtype=torch.float32)
        #
        # losses.update({
        #     "loss_unsup_u": (F.smooth_l1_loss(u_est, pseudo_u, reduction='none') * pseudo_sigma).sum() * self.w_p_points * factor,
        #     "loss_unsup_v": (F.smooth_l1_loss(v_est, pseudo_v, reduction='none') * pseudo_sigma).sum() * self.w_p_points * factor
        # })

        return losses
