# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from detectron2.layers.wrappers import _NewEmptyTensorOp


class TridentConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        paddings=0,
        dilations=1,
        groups=1,
        num_branch=1,
        test_branch_idx=-1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(TridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.dilations)}) == 1

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if inputs[0].numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    inputs[0].shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [input[0].shape[0], self.weight.shape[0]] + output_shape
            return [_NewEmptyTensorOp.apply(input, output_shape) for input in inputs]

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, self.stride, padding, dilation, self.groups)
                for input, dilation, padding in zip(inputs, self.dilations, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.stride,
                    self.paddings[self.test_branch_idx],
                    self.dilations[self.test_branch_idx],
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", num_branch=" + str(self.num_branch)
        tmpstr += ", test_branch_idx=" + str(self.test_branch_idx)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", paddings=" + str(self.paddings)
        tmpstr += ", dilations=" + str(self.dilations)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr
