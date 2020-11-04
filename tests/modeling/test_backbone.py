# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

from detectron2.config import get_cfg
from detectron2.export.torchscript_patch import patch_nonscriptable_classes
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import build_resnet_backbone


class TestBackBone(unittest.TestCase):
    def test_resnet_scriptability(self):
        patch_nonscriptable_classes()
        cfg = get_cfg()
        resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))

        scripted_resnet = torch.jit.script(resnet)

        inp = torch.rand(2, 3, 100, 100)
        out1 = resnet(inp)["res4"]
        out2 = scripted_resnet(inp)["res4"]
        self.assertTrue(torch.allclose(out1, out2))
