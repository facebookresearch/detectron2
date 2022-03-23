# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ...structures import DensePoseEmbeddingPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseEmbeddingPredictor(nn.Module):
    """
    Last layers of a DensePose model that take DensePose head outputs as an input
    and produce model outputs for continuous surface embeddings (CSE).
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize predictor using configuration options

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): input tensor size along the channel dimension
        """
        super().__init__()
        dim_in = input_channels
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        embed_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        # coarse segmentation
        self.coarse_segm_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # embedding
        self.embed_lowres = ConvTranspose2d(
            dim_in, embed_size, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def interp2d(self, tensor_nchw: torch.Tensor):
        """
        Bilinear interpolation method to be used for upscaling

        Args:
            tensor_nchw (tensor): tensor of shape (N, C, H, W)
        Return:
            tensor of shape (N, C, Hout, Wout), where Hout and Wout are computed
                by applying the scale factor to H and W
        """
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def forward(self, head_outputs):
        """
        Perform forward step on DensePose head outputs

        Args:
            head_outputs (tensor): DensePose head outputs, tensor of shape [N, D, H, W]
        """
        embed_lowres = self.embed_lowres(head_outputs)
        coarse_segm_lowres = self.coarse_segm_lowres(head_outputs)
        embed = self.interp2d(embed_lowres)
        coarse_segm = self.interp2d(coarse_segm_lowres)
        return DensePoseEmbeddingPredictorOutput(embedding=embed, coarse_segm=coarse_segm)
