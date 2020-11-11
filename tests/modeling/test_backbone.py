# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.torchscript_patch import patch_nonscriptable_classes
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.utils.env import TORCH_VERSION


class TestBackBone(unittest.TestCase):
    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_resnet_scriptability(self):
        patch_nonscriptable_classes()
        cfg = get_cfg()
        resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))

        scripted_resnet = torch.jit.script(resnet)

        inp = torch.rand(2, 3, 100, 100)
        out1 = resnet(inp)["res4"]
        out2 = scripted_resnet(inp)["res4"]
        self.assertTrue(torch.allclose(out1, out2))

    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_fpn_scriptability(self):
        patch_nonscriptable_classes()
        cfg = model_zoo.get_config("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
        bb = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        bb_s = torch.jit.script(bb)

        inp = torch.rand(2, 3, 128, 128)
        out1 = bb(inp)["p5"]
        out2 = bb_s(inp)["p5"]
        self.assertTrue(torch.allclose(out1, out2))
