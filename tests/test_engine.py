# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest
import torch
from torch import nn

from detectron2.engine import SimpleTrainer


class SimpleModel(nn.Sequential):
    def forward(self, x):
        return {"loss": x.sum() + sum([x.mean() for x in self.parameters()])}


class TestTrainer(unittest.TestCase):
    def test_simple_trainer(self, device="cpu"):
        device = torch.device(device)
        model = SimpleModel(nn.Linear(10, 10)).to(device)

        def data_loader():
            while True:
                yield torch.rand(3, 3).to(device)

        trainer = SimpleTrainer(model, data_loader(), torch.optim.SGD(model.parameters(), 0.1))
        trainer.train(0, 10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_trainer_cuda(self):
        self.test_simple_trainer(device="cuda")
