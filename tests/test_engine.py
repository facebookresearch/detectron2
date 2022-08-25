# Copyright (c) Facebook, Inc. and its affiliates.

import json
import math
import os
import tempfile
import time
import unittest
from unittest import mock
import torch
from fvcore.common.checkpoint import Checkpointer
from torch import nn

from detectron2 import model_zoo
from detectron2.config import configurable, get_cfg
from detectron2.engine import DefaultTrainer, SimpleTrainer, default_setup, hooks
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

    def test_simple_trainer_reset_dataloader(self, device="cpu"):
        model = _SimpleModel().to(device=device)
        trainer = SimpleTrainer(
            model, self._data_loader(device), torch.optim.SGD(model.parameters(), 0.1)
        )
        trainer.train(0, 10)
        trainer.reset_data_loader(lambda: self._data_loader(device))
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

    def test_default_trainer(self):
        # TODO: this test requires manifold access, so changed device to CPU. see: T88318502
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.META_ARCHITECTURE = "_SimpleModel"
        cfg.DATASETS.TRAIN = ("coco_2017_val_100",)
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            cfg.OUTPUT_DIR = d
            trainer = DefaultTrainer(cfg)

            # test property
            self.assertIs(trainer.model, trainer._trainer.model)
            trainer.model = _SimpleModel()
            self.assertIs(trainer.model, trainer._trainer.model)

    def test_checkpoint_resume(self):
        model = _SimpleModel()
        dataloader = self._data_loader("cpu")
        opt = torch.optim.SGD(model.parameters(), 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 3)

        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            trainer = SimpleTrainer(model, dataloader, opt)
            checkpointer = Checkpointer(model, d, opt=opt, trainer=trainer)

            trainer.register_hooks(
                [
                    hooks.LRScheduler(scheduler=scheduler),
                    # checkpoint after scheduler to properly save the state of scheduler
                    hooks.PeriodicCheckpointer(checkpointer, 10),
                ]
            )

            trainer.train(0, 12)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-5)
            self.assertEqual(scheduler.last_epoch, 12)
            del trainer

            opt = torch.optim.SGD(model.parameters(), 999)  # lr will be loaded
            trainer = SimpleTrainer(model, dataloader, opt)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 3)
            trainer.register_hooks(
                [
                    hooks.LRScheduler(scheduler=scheduler),
                ]
            )
            checkpointer = Checkpointer(model, d, opt=opt, trainer=trainer)
            checkpointer.resume_or_load("non_exist.pth")
            self.assertEqual(trainer.iter, 11)  # last finished iter number (0-based in Trainer)
            # number of times `scheduler.step()` was called (1-based)
            self.assertEqual(scheduler.last_epoch, 12)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-5)

    def test_eval_hook(self):
        model = _SimpleModel()
        dataloader = self._data_loader("cpu")
        opt = torch.optim.SGD(model.parameters(), 0.1)

        for total_iter, period, eval_count in [(30, 15, 2), (31, 15, 3), (20, 0, 1)]:
            test_func = mock.Mock(return_value={"metric": 3.0})
            trainer = SimpleTrainer(model, dataloader, opt)
            trainer.register_hooks([hooks.EvalHook(period, test_func)])
            trainer.train(0, total_iter)
            self.assertEqual(test_func.call_count, eval_count)

    def test_best_checkpointer(self):
        model = _SimpleModel()
        dataloader = self._data_loader("cpu")
        opt = torch.optim.SGD(model.parameters(), 0.1)
        metric_name = "metric"
        total_iter = 40
        test_period = 10
        test_cases = [
            ("max", iter([0.3, 0.4, 0.35, 0.5]), 3),
            ("min", iter([1.0, 0.8, 0.9, 0.9]), 2),
            ("min", iter([math.nan, 0.8, 0.9, 0.9]), 1),
        ]
        for mode, metrics, call_count in test_cases:
            trainer = SimpleTrainer(model, dataloader, opt)
            with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
                checkpointer = Checkpointer(model, d, opt=opt, trainer=trainer)
                trainer.register_hooks(
                    [
                        hooks.EvalHook(test_period, lambda: {metric_name: next(metrics)}),
                        hooks.BestCheckpointer(test_period, checkpointer, metric_name, mode=mode),
                    ]
                )
                with mock.patch.object(checkpointer, "save") as mock_save_method:
                    trainer.train(0, total_iter)
                    self.assertEqual(mock_save_method.call_count, call_count)

    def test_setup_config(self):
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            cfg = get_cfg()
            cfg.OUTPUT_DIR = os.path.join(d, "yacs")
            default_setup(cfg, {})

            cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py")
            cfg.train.output_dir = os.path.join(d, "omegaconf")
            default_setup(cfg, {})
