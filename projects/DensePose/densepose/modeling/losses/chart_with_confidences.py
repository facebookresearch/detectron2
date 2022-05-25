# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import Any, List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .chart import DensePoseChartLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import BilinearInterpolationHelper, LossDict


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartWithConfidenceLoss(DensePoseChartLoss):
    """ """

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
            self.uv_loss_with_confidences = IIDIsotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )
        elif self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.INDEP_ANISO:
            self.uv_loss_with_confidences = IndepAnisotropicGaussianUVLoss(
                self.confidence_model_cfg.uv_confidence.epsilon
            )

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Overrides fake losses for fine segmentation and U/V coordinates to
        include computation graphs for additional confidence parameters.
        These are used when no suitable ground truth data was found in a batch.
        The loss has a value 0 and is primarily used to construct the computation graph,
        so that `DistributedDataParallel` has similar graphs on all GPUs and can
        perform reduction properly.

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
        """
        conf_type = self.confidence_model_cfg.uv_confidence.type
        if self.confidence_model_cfg.uv_confidence.enabled:
            loss_uv = (
                densepose_predictor_outputs.u.sum() + densepose_predictor_outputs.v.sum()
            ) * 0
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                loss_uv += densepose_predictor_outputs.sigma_2.sum() * 0
            elif conf_type == DensePoseUVConfidenceType.INDEP_ANISO:
                loss_uv += (
                    densepose_predictor_outputs.sigma_2.sum()
                    + densepose_predictor_outputs.kappa_u.sum()
                    + densepose_predictor_outputs.kappa_v.sum()
                ) * 0
            return {"loss_densepose_UV": loss_uv}
        else:
            return super().produce_fake_densepose_losses_uv(densepose_predictor_outputs)

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        conf_type = self.confidence_model_cfg.uv_confidence.type
        if self.confidence_model_cfg.uv_confidence.enabled:
            u_gt = packed_annotations.u_gt[j_valid_fg]
            u_est = interpolator.extract_at_points(densepose_predictor_outputs.u)[j_valid_fg]
            v_gt = packed_annotations.v_gt[j_valid_fg]
            v_est = interpolator.extract_at_points(densepose_predictor_outputs.v)[j_valid_fg]
            sigma_2_est = interpolator.extract_at_points(densepose_predictor_outputs.sigma_2)[
                j_valid_fg
            ]
            if conf_type == DensePoseUVConfidenceType.IID_ISO:
                return {
                    "loss_densepose_UV": (
                        self.uv_loss_with_confidences(u_est, v_est, sigma_2_est, u_gt, v_gt)
                        * self.w_points
                    )
                }
            elif conf_type in [DensePoseUVConfidenceType.INDEP_ANISO]:
                kappa_u_est = interpolator.extract_at_points(densepose_predictor_outputs.kappa_u)[
                    j_valid_fg
                ]
                kappa_v_est = interpolator.extract_at_points(densepose_predictor_outputs.kappa_v)[
                    j_valid_fg
                ]
                return {
                    "loss_densepose_UV": (
                        self.uv_loss_with_confidences(
                            u_est, v_est, sigma_2_est, kappa_u_est, kappa_v_est, u_gt, v_gt
                        )
                        * self.w_points
                    )
                }
        return super().produce_densepose_losses_uv(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,
        )


class IIDIsotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of iid residuals with isotropic covariance:
    $Sigma_i = sigma_i^2 I$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi) + 2 log sigma_i^2 + ||delta_i||^2 / sigma_i^2)$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: float):
        super(IIDIsotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sigma_u: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ):
        # compute $\sigma_i^2$
        # use sigma_lower_bound to avoid degenerate solution for variance
        # (sigma -> 0)
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        # compute \|delta_i\|^2
        delta_t_delta = (u - target_u) ** 2 + (v - target_v) ** 2
        # the total loss from the formula above:
        loss = 0.5 * (self.log2pi + 2 * torch.log(sigma2) + delta_t_delta / sigma2)
        # pyre-fixme[16]: `float` has no attribute `sum`.
        return loss.sum()


class IndepAnisotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of independent residuals with anisotropic covariances:
    $Sigma_i = sigma_i^2 I + r_i r_i^T$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi)
      + log sigma_i^2 (sigma_i^2 + ||r_i||^2)
      + ||delta_i||^2 / sigma_i^2
      - <delta_i, r_i>^2 / (sigma_i^2 * (sigma_i^2 + ||r_i||^2)))$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: float):
        super(IndepAnisotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        sigma_u: torch.Tensor,
        kappa_u_est: torch.Tensor,
        kappa_v_est: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ):
        # compute $\sigma_i^2$
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        # compute \|r_i\|^2
        r_sqnorm2 = kappa_u_est**2 + kappa_v_est**2
        delta_u = u - target_u
        delta_v = v - target_v
        # compute \|delta_i\|^2
        delta_sqnorm = delta_u**2 + delta_v**2
        delta_u_r_u = delta_u * kappa_u_est
        delta_v_r_v = delta_v * kappa_v_est
        # compute the scalar product <delta_i, r_i>
        delta_r = delta_u_r_u + delta_v_r_v
        # compute squared scalar product <delta_i, r_i>^2
        delta_r_sqnorm = delta_r**2
        denom2 = sigma2 * (sigma2 + r_sqnorm2)
        loss = 0.5 * (
            self.log2pi + torch.log(denom2) + delta_sqnorm / sigma2 - delta_r_sqnorm / denom2
        )
        return loss.sum()  # pyre-ignore[16]
