# Copyright (c) Facebook, Inc. and its affiliates.
import tempfile
import time
import unittest
import torch
from torch import nn

from detectron2.config import configurable, get_cfg
from detectron2.engine import DefaultTrainer, SimpleTrainer, hooks
from detectron2.utils.events import WandbWriter
from detectron2.evaluation.wandb import WandbVisualizer
from detectron2 import model_zoo

import wandb

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


class TestWandb(unittest.TestCase):
    def _data_loader(self, device):
        device = torch.device(device)
        while True:
            yield torch.rand(3, 3).to(device)


    def test_WandbWriter(self):
        model = _SimpleModel(sleep_sec=0.1)
        trainer = SimpleTrainer(
            model, self._data_loader("cpu"), torch.optim.SGD(model.parameters(), 0.1)
        )
        max_iter = 50
        wandb_run = wandb.init(project="ci", anonymous='must')
        run_id = wandb_run.id

        trainer.register_hooks(
            [hooks.PeriodicWriter([WandbWriter()])]
        )
        trainer.train(0, max_iter)
        api = wandb.Api()
        run = api.run("ci/"+run_id)
        assert run.summary["total_loss"] # test if metric was logged

    def test_WandbVisualizer(self):

        # WandbVisualizer is more efficient is it isn't re-initialized. It reuses the references of logged tables
        wandb_run = wandb.init(project="ci", anonymous='must')
        run_id = wandb_run.id

        visualizer = WandbVisualizer("coco_2017_val_100")
        class CustomTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                return [visualizer]

        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))

        cfg.DATASETS.TRAIN = ("coco_2017_val_100",)
        cfg.DATASETS.TEST = ("coco_2017_val_100",)
        cfg.SOLVER.MAX_ITER = 3
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            cfg.OUTPUT_DIR = d
            trainer = CustomTrainer(cfg)
            trainer.train()
        
        api = wandb.Api()
        run = api.run("ci/"+run_id)
        assert run.summary["coco_2017_val_100"] # test is evaluation table was logged
