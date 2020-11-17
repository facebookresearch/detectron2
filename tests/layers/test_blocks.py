# -*- coding: utf-8 -*-


import unittest
import torch
from torch import nn

from detectron2.layers import ASPP, DepthwiseSeparableConv2d, FrozenBatchNorm2d
from detectron2.utils.env import TORCH_VERSION


"""
Test for misc layers.
"""


class TestBlocks(unittest.TestCase):
    def test_separable_conv(self):
        DepthwiseSeparableConv2d(3, 10, norm1="BN", activation1=nn.PReLU())

    def test_aspp(self):
        m = ASPP(3, 10, [2, 3, 4], norm="", activation=nn.PReLU())
        self.assertIsNot(m.convs[0].activation.weight, m.convs[1].activation.weight)
        self.assertIsNot(m.convs[0].activation.weight, m.project.activation.weight)

    @unittest.skipIf(TORCH_VERSION < (1, 6) or not torch.cuda.is_available(), "CUDA not available")
    def test_frozen_batchnorm_fp16(self):
        from torch.cuda.amp import autocast

        C = 10
        input = torch.rand(1, C, 10, 10).cuda()
        m = FrozenBatchNorm2d(C).cuda()
        with autocast():
            output = m(input.half())
        self.assertEqual(output.dtype, torch.float16)

        # requires_grad triggers a different codepath
        input.requires_grad_()
        with autocast():
            output = m(input.half())
        self.assertEqual(output.dtype, torch.float16)
