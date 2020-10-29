# -*- coding: utf-8 -*-


import unittest
from torch import nn

from detectron2.layers import ASPP, DepthwiseSeparableConv2d


class TestBlocks(unittest.TestCase):
    def test_separable_conv(self):
        DepthwiseSeparableConv2d(3, 10, norm1="BN", activation1=nn.PReLU())

    def test_aspp(self):
        m = ASPP(3, 10, [2, 3, 4], norm="", activation=nn.PReLU())
        self.assertIsNot(m.convs[0].activation.weight, m.convs[1].activation.weight)
        self.assertIsNot(m.convs[0].activation.weight, m.project.activation.weight)
