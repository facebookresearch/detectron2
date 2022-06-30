# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
torch.cuda.empty_cache()

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ...structures import BlockPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseChartBlockPredictor(nn.Module):
    """
    Predictor (last layers of a DensePose model) that takes DensePose head outputs as an input
    and produces 6 tensors which represent DensePose results for predefined body parts
    (patches / charts):
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, two tensor of shape [N, Cb * C, Hout, Wout]
     * V coordinates, two tensor of shape [N, Cb * C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Cb is the number of block
     - Hout and Wout are height and width of predictions
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
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        dim_out_patches_uv = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES * cfg.MODEL.BLOCK.BLOCK_NUM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        # coarse segmentation
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # fine segmentation
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # U
        self.u_cls_lowres = ConvTranspose2d(
            dim_in, dim_out_patches_uv, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_offset_lowres = ConvTranspose2d(
            dim_in, dim_out_patches_uv, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # V
        self.v_cls_lowres = ConvTranspose2d(
            dim_in, dim_out_patches_uv, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_offset_lowres = ConvTranspose2d(
            dim_in, dim_out_patches_uv, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
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

    def forward(self, head_outputs: torch.Tensor):
        """
        Perform forward step on DensePose head outputs

        Args:
            head_outputs (tensor): DensePose head outputs, tensor of shape [N, D, H, W]
        Return:
           An instance of DensePoseChartPredictorOutput
        """
        return BlockPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=self.interp2d(self.index_uv_lowres(head_outputs)),
            u_cls=self.interp2d(self.u_cls_lowres(head_outputs)),
            u_offset=self.interp2d(self.u_offset_lowres(head_outputs)),
            v_cls=self.interp2d(self.v_cls_lowres(head_outputs)),
            v_offset=self.interp2d(self.v_offset_lowres(head_outputs))
        )