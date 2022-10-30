# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate, Conv2d

from ...structures import DensePoseChartPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseChartPredictor(nn.Module):
    """
    Predictor (last layers of a DensePose model) that takes DensePose head outputs as an input
    and produces 4 tensors which represent DensePose results for predefined body parts
    (patches / charts):
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
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
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        # V
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )

        # corrector
        hidden_dim = cfg.MODEL.SEMI.COR.CONV_HEAD_DIM
        conv_kernel_size = cfg.MODEL.SEMI.COR.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.SEMI.COR.NUM_STACKED_CONVS
        pad_size = conv_kernel_size // 2
        # n_pred_channels = (cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1) + 2
        # n_channels = n_pred_channels + 256 + 1  # + cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        n_channels = dim_in
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, conv_kernel_size, stride=1, padding=pad_size)
            layer_name = _get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        # self.n_out_channels = n_channels
        self.crt_segm = ConvTranspose2d(
            dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.uv_confidence = cfg.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.ENABLED
        if self.uv_confidence:
            self.crt_sigma = ConvTranspose2d(
                dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )

        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)
        self.non_local = NonLocalBlock(in_channels=n_channels)

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
        fine_segm = self.interp2d(self.index_uv_lowres(head_outputs))

        crt_output = head_outputs.detach()
        crt_output = self.non_local(crt_output)
        for i in range(self.n_stacked_convs):
            layer_name = _get_layer_name(i)
            crt_output = getattr(self, layer_name)(crt_output)
            crt_output = F.relu(crt_output)

        output = DensePoseChartPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=fine_segm,
            u=self.interp2d(self.u_lowres(head_outputs)),
            v=self.interp2d(self.v_lowres(head_outputs)),
            crt_segm=self.interp2d(self.crt_segm(crt_output)),
            crt_sigma=self.interp2d(self.crt_sigma(crt_output)) if self.uv_confidence else None,
        )
        return output


def _get_layer_name(i: int):
    layer_name = "body_conv_fcn{}".format(i + 1)
    return layer_name


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True) -> None:
        super().__init__()
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == 'embedded' or self.mode == 'dot' or self.mode == 'concatenate':
            self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == 'concatenate':
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size = x.shape[0]

        if batch_size == 0:
            return x

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == 'gaussian':
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'embedded' or self.mode == 'dot':
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'concatenate':
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.sze(0), f.size(2), f.size(3))

        if self.mode == 'gaussian' or self.mode == 'embedded':
            f_div_C = F.softmax(f, dim=1)
        elif self.mode == 'dot' or self.mode == 'concatenate':
            N = f.size(-1)
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        z = W_y + x

        return z
