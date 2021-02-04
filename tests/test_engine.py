# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
import tempfile
import time
import unittest
import torch
from torch import nn

from detectron2.config import configurable, get_cfg
from detectron2.engine import DefaultTrainer, SimpleTrainer, hooks
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.utils.events import CommonMetricPrinter, JSONWriter


@META_ARCH_REGISTRY.register()
class _SimpleModel(nn.Module):
    @configurable
    def __init__(self, sleep_sec=0):
        super().__init__()
        self.mod = nn.Linear(10, 20)
        self.sleep_sec = sleep_sec

    @classmethod
    def from_config(cls, cfg):
        return {}

    def forward(self, x):
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        return {"loss": x.sum() + sum([x.mean() for x in self.parameters()])}


class TestTrainer(unittest.TestCase):
    def _data_loader(self, device):
        device = torch.device(device)
        while True:
            yield torch.rand(3, 3).to(device)

    def test_simple_trainer(self, device="cpu"):
        model = _SimpleModel().to(device=device)
        trainer = SimpleTrainer(
            model, self._data_loader(device), torch.optim.SGD(model.parameters(), 0.1)
        )
        trainer.train(0, 10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_trainer_cuda(self):
        self.test_simple_trainer(device="cuda")

    def test_writer_hooks(self):
        model = _SimpleModel(sleep_sec=0.1)
        trainer = SimpleTrainer(
            model, self._data_loader("cpu"), torch.optim.SGD(model.parameters(), 0.1)
        )

        max_iter = 50

        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            json_file = os.path.join(d, "metrics.json")
            writers = [CommonMetricPrinter(max_iter), JSONWriter(json_file)]

            trainer.register_hooks(
                [hooks.EvalHook(0, lambda: {"metric": 100}), hooks.PeriodicWriter(writers)]
            )
            with self.assertLogs(writers[0].logger) as logs:
                trainer.train(0, max_iter)

            with open(json_file, "r") as f:
                data = [json.loads(line.strip()) for line in f]
                self.assertEqual([x["iteration"] for x in data], [19, 39, 49, 50])
                # the eval metric is in the last line with iter 50
                self.assertIn("metric", data[-1], "Eval metric must be in last line of JSON!")

            # test logged messages from CommonMetricPrinter
            self.assertEqual(len(logs.output), 3)
            for log, iter in zip(logs.output, [19, 39, 49]):
                self.assertIn(f"iter: {iter}", log)

            self.assertIn("eta: 0:00:00", logs.output[-1], "Last ETA must be 0!")

    @unittest.skipIf(os.environ.get("CI"), "Require COCO data.")
    def test_default_trainer(self):
        cfg = get_cfg()
        cfg.MODEL.META_ARCHITECTURE = "_SimpleModel"
        cfg.DATASETS.TRAIN = ("coco_2017_val_100",)
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            cfg.OUTPUT_DIR = d
            trainer = DefaultTrainer(cfg)

            # test property
            self.assertIs(trainer.model, trainer._trainer.model)
            trainer.model = _SimpleModel()
            self.assertIs(trainer.model, trainer._trainer.model)
