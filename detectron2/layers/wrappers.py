# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torch.nn.modules.utils import _ntuple

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if TORCH_VERSION > (1, 4):
    ConvTranspose2d = torch.nn.ConvTranspose2d
else:

    class ConvTranspose2d(torch.nn.ConvTranspose2d):
        """
        A wrapper around :class:`torch.nn.ConvTranspose2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(ConvTranspose2d, self).forward(x)
            # get output shape

            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
                for i, p, di, k, d, op in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                    self.output_padding,
                )
            ]
            output_shape = [x.shape[0], self.out_channels] + output_shape
            # This is to make DDP happy.
            # DDP expects all workers to have gradient w.r.t the same set of parameters.
            _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
            return _NewEmptyTensorOp.apply(x, output_shape) + _dummy


if TORCH_VERSION > (1, 4):
    BatchNorm2d = torch.nn.BatchNorm2d
else:

    class BatchNorm2d(torch.nn.BatchNorm2d):
        """
        A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(BatchNorm2d, self).forward(x)
            # get output shape
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)


if TORCH_VERSION > (1, 5):
    Linear = torch.nn.Linear
else:

    class Linear(torch.nn.Linear):
        """
        A wrapper around :class:`torch.nn.Linear` to support empty inputs and more features.
        Because of https://github.com/pytorch/pytorch/issues/34202
        """

        def forward(self, x):
            if x.numel() == 0:
                output_shape = [x.shape[0], self.weight.shape[0]]

                empty = _NewEmptyTensorOp.apply(x, output_shape)
                if self.training:
                    # This is to make DDP happy.
                    # DDP expects all workers to have gradient w.r.t the same set of parameters.
                    _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                    return empty + _dummy
                else:
                    return empty

            x = super().forward(x)
            return x


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    A wrapper around :func:`torch.nn.functional.interpolate` to support zero-size tensor.
    """
    if TORCH_VERSION > (1, 4) or input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners=align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
