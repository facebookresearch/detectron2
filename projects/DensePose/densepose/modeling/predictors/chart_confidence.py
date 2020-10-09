# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d

from ..confidence import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from ..utils import initialize_module_params


class DensePoseChartConfidencePredictorMixin:
    """
    Predictor contains the last layers of a DensePose model that take DensePose head
    outputs as an input and produce model outputs. Confidence predictor mixin is used
    to generate confidences for segmentation and UV tensors estimated by some
    base predictor. Several assumptions need to hold for the base predictor:
    1) the `forward` method must return SIUV tuple as the first result (
        S = coarse segmentation, I = fine segmentation, U and V are intrinsic
        chart coordinates)
    2) `interp2d` method must be defined to perform bilinear interpolation;
        the same method is typically used for SIUV and confidences
    Confidence predictor mixin provides confidence estimates, as described in:
        N. Neverova et al., Correlated Uncertainty for Learning Dense Correspondences
            from Noisy Labels, NeurIPS 2019
        A. Sanakoyeu et al., Transferring Dense Pose to Proximal Animal Classes, CVPR 2020
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize confidence predictor using configuration options.

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        # we rely on base predictor to call nn.Module.__init__
        super().__init__(cfg, input_channels)
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        self._initialize_confidence_estimation_layers(cfg, input_channels)
        initialize_module_params(self)

    def _initialize_confidence_estimation_layers(self, cfg: CfgNode, dim_in: int):
        """
        Initialize confidence estimation layers based on configuration options

        Args:
            cfg (CfgNode): configuration options
            dim_in (int): number of input channels
        """
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        if self.confidence_model_cfg.uv_confidence.enabled:
            if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
                self.sigma_2_lowres = ConvTranspose2d(
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
            elif (
                self.confidence_model_cfg.uv_confidence.type
                == DensePoseUVConfidenceType.INDEP_ANISO
            ):
                self.sigma_2_lowres = ConvTranspose2d(
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
                self.kappa_u_lowres = ConvTranspose2d(
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
                self.kappa_v_lowres = ConvTranspose2d(
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
            else:
                raise ValueError(
                    f"Unknown confidence model type: "
                    f"{self.confidence_model_cfg.confidence_model_type}"
                )
        if self.confidence_model_cfg.segm_confidence.enabled:
            self.fine_segm_confidence_lowres = ConvTranspose2d(
                dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )
            self.coarse_segm_confidence_lowres = ConvTranspose2d(
                dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )

    def forward(self, head_outputs: torch.Tensor):
        """
        Perform forward operation on head outputs used as inputs for the predictor.
        Calls forward method from the base predictor and uses its outputs to compute
        confidences.

        Args:
            head_outputs (Tensor): head outputs used as predictor inputs
        Return:
            A tuple containing the following entries:
            - SIUV tuple with possibly modified segmentation tensors
            - various other outputs from the base predictor
            - 6 tensors with estimated confidence model parameters at full resolution
            (sigma_1, sigma_2, kappa_u, kappa_v, fine_segm_confidence, coarse_segm_confidence)
            - 6 tensors with estimated confidence model parameters at half resolution
            (sigma_1, sigma_2, kappa_u, kappa_v, fine_segm_confidence, coarse_segm_confidence)
        """
        # assuming base class returns SIUV estimates in its first result
        base_predictor_outputs = super().forward(head_outputs)
        siuv = (
            base_predictor_outputs[0]
            if isinstance(base_predictor_outputs, tuple)
            else base_predictor_outputs
        )
        coarse_segm, fine_segm, u, v = siuv

        sigma_1, sigma_2, kappa_u, kappa_v = None, None, None, None
        sigma_1_lowres, sigma_2_lowres, kappa_u_lowres, kappa_v_lowres = None, None, None, None
        fine_segm_confidence_lowres, fine_segm_confidence = None, None
        coarse_segm_confidence_lowres, coarse_segm_confidence = None, None
        if self.confidence_model_cfg.uv_confidence.enabled:
            if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
                sigma_2_lowres = self.sigma_2_lowres(head_outputs)
                # assuming base class defines interp2d method for bilinear interpolation
                sigma_2 = self.interp2d(sigma_2_lowres)
            elif (
                self.confidence_model_cfg.uv_confidence.type
                == DensePoseUVConfidenceType.INDEP_ANISO
            ):
                sigma_2_lowres = self.sigma_2_lowres(head_outputs)
                kappa_u_lowres = self.kappa_u_lowres(head_outputs)
                kappa_v_lowres = self.kappa_v_lowres(head_outputs)
                # assuming base class defines interp2d method for bilinear interpolation
                sigma_2 = self.interp2d(sigma_2_lowres)
                kappa_u = self.interp2d(kappa_u_lowres)
                kappa_v = self.interp2d(kappa_v_lowres)
            else:
                raise ValueError(
                    f"Unknown confidence model type: "
                    f"{self.confidence_model_cfg.confidence_model_type}"
                )
        if self.confidence_model_cfg.segm_confidence.enabled:
            fine_segm_confidence_lowres = self.fine_segm_confidence_lowres(head_outputs)
            # assuming base class defines interp2d method for bilinear interpolation
            fine_segm_confidence = self.interp2d(fine_segm_confidence_lowres)
            fine_segm_confidence = (
                F.softplus(fine_segm_confidence) + self.confidence_model_cfg.segm_confidence.epsilon
            )
            fine_segm = fine_segm * torch.repeat_interleave(
                fine_segm_confidence, fine_segm.shape[1], dim=1
            )
            coarse_segm_confidence_lowres = self.coarse_segm_confidence_lowres(head_outputs)
            # assuming base class defines interp2d method for bilinear interpolation
            coarse_segm_confidence = self.interp2d(coarse_segm_confidence_lowres)
            coarse_segm_confidence = (
                F.softplus(coarse_segm_confidence)
                + self.confidence_model_cfg.segm_confidence.epsilon
            )
            coarse_segm = coarse_segm * torch.repeat_interleave(
                coarse_segm_confidence, coarse_segm.shape[1], dim=1
            )
        results = []
        # append SIUV with possibly modified segmentation tensors
        results.append((coarse_segm, fine_segm, u, v))
        # append the rest of base predictor outputs
        if isinstance(base_predictor_outputs, tuple):
            results.extend(base_predictor_outputs[1:])
        # append hi-res confidence estimates
        results.append(
            (sigma_1, sigma_2, kappa_u, kappa_v, fine_segm_confidence, coarse_segm_confidence)
        )
        # append lo-res confidence estimates
        results.append(
            (
                sigma_1_lowres,
                sigma_2_lowres,
                kappa_u_lowres,
                kappa_v_lowres,
                fine_segm_confidence_lowres,
                coarse_segm_confidence_lowres,
            )
        )
        return tuple(results)
