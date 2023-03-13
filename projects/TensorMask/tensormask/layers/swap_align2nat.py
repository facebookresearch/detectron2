# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from tensormask import _C


class _SwapAlign2Nat(Function):
    @staticmethod
    def forward(ctx, X, lambda_val, pad_val):
        ctx.lambda_val = lambda_val
        ctx.input_shape = X.size()

        Y = _C.swap_align2nat_forward(X, lambda_val, pad_val)
        return Y

    @staticmethod
    @once_differentiable
    def backward(ctx, gY):
        lambda_val = ctx.lambda_val
        bs, ch, h, w = ctx.input_shape

        gX = _C.swap_align2nat_backward(gY, lambda_val, bs, ch, h, w)

        return gX, None, None


swap_align2nat = _SwapAlign2Nat.apply


class SwapAlign2Nat(nn.Module):
    """
    The op `SwapAlign2Nat` described in https://arxiv.org/abs/1903.12174.
    Given an input tensor that predicts masks of shape (N, C=VxU, H, W),
    apply the op, it will return masks of shape (N, V'xU', H', W') where
    the unit lengths of (V, U) and (H, W) are swapped, and the mask representation
    is transformed from aligned to natural.
    Args:
        lambda_val (int): the relative unit length ratio between (V, U) and (H, W),
                            as we always have larger unit lengths for (V, U) than (H, W),
                            lambda_val is always >= 1.
        pad_val (float):    padding value for the values falling outside of the input
                            tensor, default set to -6 as sigmoid(-6) is ~0, indicating
                            that is no masks outside of the tensor.
    """

    def __init__(self, lambda_val, pad_val=-6.0):
        super(SwapAlign2Nat, self).__init__()
        self.lambda_val = lambda_val
        self.pad_val = pad_val

    def forward(self, X):
        return swap_align2nat(X, self.lambda_val, self.pad_val)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "lambda_val=" + str(self.lambda_val)
        tmpstr += ", pad_val=" + str(self.pad_val)
        tmpstr += ")"
        return tmpstr
