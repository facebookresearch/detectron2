# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

from typing import List
import torch
from torch.nn import functional as F

from detectron2.utils.env import TORCH_VERSION


def cat(tensors: List[torch.Tensor], dim: int = 0):
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
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate


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


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)
