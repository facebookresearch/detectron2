# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d

from densepose.modeling.confidence import DensePoseConfidenceModelConfig
from densepose.modeling.utils import initialize_module_params
from densepose.structures import decorate_cse_predictor_output_class_with_confidences


class DensePoseEmbeddingConfidencePredictorMixin:
    """
    Predictor contains the last layers of a DensePose model that take DensePose head
    outputs as an input and produce model outputs. Confidence predictor mixin is used
    to generate confidences for coarse segmentation estimated by some
    base predictor. Several assumptions need to hold for the base predictor:
    1) the `forward` method must return CSE DensePose head outputs,
        tensor of shape [N, D, H, W]
    2) `interp2d` method must be defined to perform bilinear interpolation;
        the same method is typically used for masks and confidences
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
        super().__init__(cfg, input_channels)  # pyre-ignore[19]
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        self._initialize_confidence_estimation_layers(cfg, input_channels)
        self._registry = {}
        initialize_module_params(self)  # pyre-ignore[6]

    def _initialize_confidence_estimation_layers(self, cfg: CfgNode, dim_in: int):
        """
        Initialize confidence estimation layers based on configuration options

        Args:
            cfg (CfgNode): configuration options
            dim_in (int): number of input channels
        """
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        if self.confidence_model_cfg.segm_confidence.enabled:
            self.coarse_segm_confidence_lowres = ConvTranspose2d(  # pyre-ignore[16]
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
            An instance of outputs with confidences,
            see `decorate_cse_predictor_output_class_with_confidences`
        """
        # assuming base class returns SIUV estimates in its first result
        base_predictor_outputs = super().forward(head_outputs)  # pyre-ignore[16]

        # create output instance by extending base predictor outputs:
        output = self._create_output_instance(base_predictor_outputs)

        if self.confidence_model_cfg.segm_confidence.enabled:
            # base predictor outputs are assumed to have `coarse_segm` attribute
            # base predictor is assumed to define `interp2d` method for bilinear interpolation
            output.coarse_segm_confidence = (
                F.softplus(
                    self.interp2d(  # pyre-ignore[16]
                        self.coarse_segm_confidence_lowres(head_outputs)  # pyre-ignore[16]
                    )
                )
                + self.confidence_model_cfg.segm_confidence.epsilon
            )
            output.coarse_segm = base_predictor_outputs.coarse_segm * torch.repeat_interleave(
                output.coarse_segm_confidence, base_predictor_outputs.coarse_segm.shape[1], dim=1
            )

        return output

    def _create_output_instance(self, base_predictor_outputs: Any):
        """
        Create an instance of predictor outputs by copying the outputs from the
        base predictor and initializing confidence

        Args:
            base_predictor_outputs: an instance of base predictor outputs
                (the outputs type is assumed to be a dataclass)
        Return:
           An instance of outputs with confidences
        """
        PredictorOutput = decorate_cse_predictor_output_class_with_confidences(
            type(base_predictor_outputs)  # pyre-ignore[6]
        )
        # base_predictor_outputs is assumed to be a dataclass
        # reassign all the fields from base_predictor_outputs (no deep copy!), add new fields
        output = PredictorOutput(
            **base_predictor_outputs.__dict__,
            coarse_segm_confidence=None,
        )
        return output
