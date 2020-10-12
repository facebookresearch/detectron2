# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F

from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .chart import extract_data_for_mask_loss_from_matches
from .chart_with_confidences import IIDIsotropicGaussianUVLoss, IndepAnisotropicGaussianUVLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import BilinearInterpolationHelper, SingleTensorsHelper, resample_data


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
            self.uv_loss_with_confidences = IIDIsotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )
        elif self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.INDEP_ANISO:
            self.uv_loss_with_confidences = IndepAnisotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )

    def __call__(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        if not self.segm_trained_by_masks:
            return self.produce_densepose_losses(
                proposals_with_gt, densepose_outputs, densepose_confidences
            )
        else:
            losses = {}
            losses_densepose = self.produce_densepose_losses(
                proposals_with_gt, densepose_outputs, densepose_confidences
            )
            losses.update(losses_densepose)
            losses_mask = self.produce_mask_losses(
                proposals_with_gt, densepose_outputs, densepose_confidences
            )
            losses.update(losses_mask)
            return losses

    def produce_fake_mask_losses(self, densepose_outputs):
        losses = {}
        segm_scores, _, _, _ = densepose_outputs
        losses["loss_densepose_S"] = segm_scores.sum() * 0
        return losses

    def produce_mask_losses(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        if not len(proposals_with_gt):
            return self.produce_fake_mask_losses(densepose_outputs)
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        segm_scores, _, _, _ = densepose_outputs
        with torch.no_grad():
            mask_loss_data = extract_data_for_mask_loss_from_matches(proposals_with_gt, segm_scores)
        if (mask_loss_data.masks_gt is None) or (mask_loss_data.masks_est is None):
            return self.produce_fake_mask_losses(densepose_outputs)
        losses["loss_densepose_S"] = (
            F.cross_entropy(mask_loss_data.masks_est, mask_loss_data.masks_gt.long()) * self.w_segm
        )
        return losses

    def produce_fake_densepose_losses(self, densepose_outputs, densepose_confidences):
        # we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) losses in the form Tensor.sum() * 0
        s, index_uv, u, v = densepose_outputs
        conf_type = self.confidence_model_cfg.uv_confidence.type
        (
            sigma_1,
            sigma_2,
            kappa_u,
            kappa_v,
            fine_segm_confidence,
            coarse_segm_confidence,
        ) = densepose_confidences
        losses = {}
        losses["loss_densepose_I"] = index_uv.sum() * 0
        if not self.segm_trained_by_masks:
            losses["loss_densepose_S"] = s.sum() * 0
        if self.confidence_model_cfg.uv_confidence.enabled:
            losses["loss_densepose_UV"] = (u.sum() + v.sum()) * 0
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                losses["loss_densepose_UV"] += sigma_2.sum() * 0
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                losses["loss_densepose_UV"] += (sigma_2.sum() + kappa_u.sum() + kappa_v.sum()) * 0
        else:
            losses["loss_densepose_U"] = u.sum() * 0
            losses["loss_densepose_V"] = v.sum() * 0
        return losses

    def produce_densepose_losses(self, proposals_with_gt, densepose_outputs, densepose_confidences):
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        s, index_uv, u, v = densepose_outputs
        assert u.size(2) == v.size(2)
        assert u.size(3) == v.size(3)
        assert u.size(2) == index_uv.size(2)
        assert u.size(3) == index_uv.size(3)
        densepose_outputs_size = u.size()

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)
        (
            sigma_1,
            sigma_2,
            kappa_u,
            kappa_v,
            fine_segm_confidence,
            coarse_segm_confidence,
        ) = densepose_confidences
        conf_type = self.confidence_model_cfg.uv_confidence.type

        tensors_helper = SingleTensorsHelper(proposals_with_gt)
        n_batch = len(tensors_helper.index_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            return self.produce_fake_densepose_losses(densepose_outputs, densepose_confidences)

        interpolator = BilinearInterpolationHelper.from_matches(
            tensors_helper, densepose_outputs_size
        )

        j_valid_fg = interpolator.j_valid * (tensors_helper.fine_segm_labels_gt > 0)

        u_gt = tensors_helper.u_gt_all[j_valid_fg]
        u_est_all = interpolator.extract_at_points(u[tensors_helper.index_with_dp])
        u_est = u_est_all[j_valid_fg]

        v_gt = tensors_helper.v_gt_all[j_valid_fg]
        v_est_all = interpolator.extract_at_points(v[tensors_helper.index_with_dp])
        v_est = v_est_all[j_valid_fg]

        index_uv_gt = tensors_helper.fine_segm_labels_gt[interpolator.j_valid]
        index_uv_est_all = interpolator.extract_at_points(
            index_uv[tensors_helper.index_with_dp],
            slice_index_uv=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )
        index_uv_est = index_uv_est_all[interpolator.j_valid, :]

        if self.confidence_model_cfg.uv_confidence.enabled:
            sigma_2_est_all = interpolator.extract_at_points(sigma_2[tensors_helper.index_with_dp])
            sigma_2_est = sigma_2_est_all[j_valid_fg]
            if conf_type in [DensePoseUVConfidenceType.INDEP_ANISO]:
                kappa_u_est_all = interpolator.extract_at_points(
                    kappa_u[tensors_helper.index_with_dp]
                )
                kappa_u_est = kappa_u_est_all[j_valid_fg]
                kappa_v_est_all = interpolator.extract_at_points(
                    kappa_v[tensors_helper.index_with_dp]
                )
                kappa_v_est = kappa_v_est_all[j_valid_fg]

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        if not self.segm_trained_by_masks:
            s_est = s[tensors_helper.index_with_dp]
            with torch.no_grad():
                s_gt = resample_data(
                    tensors_helper.coarse_segm_gt.unsqueeze(1),
                    tensors_helper.bbox_xywh_gt,
                    tensors_helper.bbox_xywh_est,
                    self.heatmap_size,
                    self.heatmap_size,
                    mode="nearest",
                    padding_mode="zeros",
                ).squeeze(1)

        # add point-based losses:
        if self.confidence_model_cfg.uv_confidence.enabled:
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                uv_loss = (
                    self.uv_loss_with_confidences(u_est, v_est, sigma_2_est, u_gt, v_gt)
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                uv_loss = (
                    self.uv_loss_with_confidences(
                        u_est, v_est, sigma_2_est, kappa_u_est, kappa_v_est, u_gt, v_gt
                    )
                    * self.w_points
                )
                losses["loss_densepose_UV"] = uv_loss
            else:
                raise ValueError(f"Unknown confidence model type: {conf_type}")
        else:
            u_loss = F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points
            losses["loss_densepose_U"] = u_loss
            v_loss = F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points
            losses["loss_densepose_V"] = v_loss
        index_uv_loss = F.cross_entropy(index_uv_est, index_uv_gt.long()) * self.w_part
        losses["loss_densepose_I"] = index_uv_loss

        if not self.segm_trained_by_masks:
            if self.n_segm_chan == 2:
                s_gt = s_gt > 0
            s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
            losses["loss_densepose_S"] = s_loss
        return losses
