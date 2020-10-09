# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ..utils import initialize_module_params


class DensePoseChartPredictor(nn.Module):
    """
    Predictor (last layers of a DensePose model) that takes DensePose head outputs as an input
    and produces 4 tensors which represent DensePose results for predefined body parts
    (patches / charts):
     - coarse segmentation [N, K, H, W]
     - fine segmentation [N, C, H, W]
     - U coordinates [N, C, H, W]
     - V coordinates [N, C, H, W]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - H and W are height and width of predictions
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
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
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
           - a tuple of 4 tensors containing DensePose predictions for charts:
               * coarse segmentation estimate, a tensor of shape [N, K, Hout, Wout]
               * fine segmentation estimate, a tensor of shape [N, C, Hout, Wout]
               * U coordinates, a tensor of shape [N, C, Hout, Wout]
               * V coordinates, a tensor of shape [N, C, Hout, Wout]
           - a tuple of 4 tensors containing DensePose predictions for charts at reduced resolution:
               * coarse segmentation estimate, a tensor of shape [N, K, Hout / 2, Wout / 2]
               * fine segmentation estimate, a tensor of shape [N, C, Hout / 2, Wout / 2]
               * U coordinates, a tensor of shape [N, C, Hout / 2, Wout / 2]
               * V coordinates, a tensor of shape [N, C, Hout / 2, Wout / 2]
        """
        coarse_segm_lowres = self.ann_index_lowres(head_outputs)
        fine_segm_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)

        coarse_segm = self.interp2d(coarse_segm_lowres)
        fine_segm = self.interp2d(fine_segm_lowres)
        u = self.interp2d(u_lowres)
        v = self.interp2d(v_lowres)
        siuv = (coarse_segm, fine_segm, u, v)
        siuv_lowres = (coarse_segm_lowres, fine_segm_lowres, u_lowres, v_lowres)
        return siuv, siuv_lowres
