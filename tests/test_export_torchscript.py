# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
import unittest
import torch
from torch import Tensor, nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.torchscript import dump_torchscript_IR, export_torchscript_with_instances
from detectron2.modeling import build_backbone
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION


@unittest.skipIf(
    os.environ.get("CIRCLECI") or TORCH_VERSION < (1, 8), "Insufficient Pytorch version"
)
class TestScripting(unittest.TestCase):
    def testMaskRCNN(self):
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    def _test_model(self, config_path, device="cpu"):
        model = model_zoo.get(config_path, trained=True, device=device)
        model.eval()

        fields = {
            "proposal_boxes": Boxes,
            "objectness_logits": Tensor,
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
            "pred_masks": Tensor,
        }
        model = export_torchscript_with_instances(model, fields)


class TestTorchscriptUtils(unittest.TestCase):
    # TODO: add test to dump scripting
    def test_dump_IR_tracing(self):
        cfg = get_cfg()
        cfg.MODEL.RESNETS.DEPTH = 18
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

        class Mod(nn.Module):
            def forward(self, x):
                return tuple(self.m(x).values())

        model = Mod()
        model.m = build_backbone(cfg)
        model.eval()

        with torch.no_grad():
            ts_model = torch.jit.trace(model, (torch.rand(2, 3, 224, 224),))

        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            dump_torchscript_IR(ts_model, d)
            # check that the files are created
            for name in ["model_ts_code", "model_ts_IR", "model_ts_IR_inlined", "model"]:
                fname = os.path.join(d, name + ".txt")
                self.assertTrue(os.stat(fname).st_size > 0, fname)
