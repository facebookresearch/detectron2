import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def checkerboard(x: torch.Tensor, drop_prob: float = 0.1, drop_size: int = 1, drop_shape: str = 'square'):
    B, C, H, W = x.shape
    total_size = W * H
    block_mask_indexes = torch.rand((1, C, 1, 1), dtype=x.dtype, device=x.device)

    if drop_shape == 'square':
        shape = round(min(W,H) / 2 * drop_size)
        # for i in indexes:
        mask = kron(torch.Tensor([[1, 0] * shape, [0, 1] * shape] * shape),
                    torch.ones((shape, shape)))[:H, :W].to(x.device)

    if drop_shape == 'rectangle':
        shape = round(min(W,H) / 2 * self.drop_size)
        ver, hor = torch.rand(2)
        # for i in indexes:
        if ver > hor:
            mask = kron(torch.Tensor([[1, 0] * int((shape / 2)), [0, 1] * int((shape / 2))] * 2 * shape),
                        torch.ones((shape, 2 * shape)))[:H, :W].to(x.device)
        else:
            mask = kron(torch.Tensor([[1, 0] * 2 * shape, [0, 1] * 2 * shape] * int((shape / 2))),
                        torch.ones((2 * shape, shape)))[:H, :W].to(x.device)
            
    # normalize_scale = (mask.numel() / mask.to(dtype=torch.float32).sum()).to(dtype=x.dtype)
    x = torch.where(block_mask_indexes<drop_prob, x*mask.to(dtype=x.dtype), x)

    return x
    

class Checkerboard(nn.Module):
    def __init__(self, drop_prob, drop_size, drop_shape):
        super(Checkerboard, self).__init__()

        self.drop_prob = drop_prob
        self.drop_size = drop_size
        self.drop_shape = drop_shape

    def forward(self, x):
        # shape: (bsize, channels, H, W)
        
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (B, C, H, W)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            return checkerboard(
                x, self.drop_prob, self.drop_size, self.drop_shape)