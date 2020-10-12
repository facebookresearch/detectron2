# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest
import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.export.torchscript import dump_torchscript_IR
from detectron2.modeling import build_backbone


class TestTorchscript(unittest.TestCase):
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
