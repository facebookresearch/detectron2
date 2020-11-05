# Copyright (c) Facebook, Inc. and its affiliates.

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d

from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseDeepLabHead(nn.Module):
    """
    DensePose head using DeepLabV3 model from
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>.
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        super(DensePoseDeepLabHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        self.use_nonlocal    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels

        self.ASPP = ASPP(input_channels, [6, 12, 56], n_channels)  # 6, 12, 56
        self.add_module("ASPP", self.ASPP)

        if self.use_nonlocal:
            self.NLBlock = NONLocalBlock2D(input_channels, bn_layer=True)
            self.add_module("NLBlock", self.NLBlock)
        # weight_init.c2_msra_fill(self.ASPP)

        for i in range(self.n_stacked_convs):
            norm_module = nn.GroupNorm(32, hidden_dim) if norm == "GN" else None
            layer = Conv2d(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                padding=pad_size,
                bias=not norm,
                norm=norm_module,
            )
            weight_init.c2_msra_fill(layer)
            n_channels = hidden_dim
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
        self.n_out_channels = hidden_dim
        # initialize_module_params(self)

    def forward(self, features):
        x0 = features
        x = self.ASPP(x0)
        if self.use_nonlocal:
            x = self.NLBlock(x)
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


# Copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# See https://arxiv.org/pdf/1706.05587.pdf for details
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# copied from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py
# See https://arxiv.org/abs/1711.07971 for details
class _NonLocalBlockND(nn.Module):
    def __init__(
        self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim) #nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(32, self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )
