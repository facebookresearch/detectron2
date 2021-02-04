# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np
from unittest import TestCase
import torch
from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    CosineParamScheduler,
    LinearParamScheduler,
    MultiStepParamScheduler,
)
from torch import nn

from detectron2.solver import LRMultiplier


class TestScheduler(TestCase):
    def test_warmup_multistep(self):
        p = nn.Parameter(torch.zeros(0))
        opt = torch.optim.SGD([p], lr=5)

        multiplier = CompositeParamScheduler(
            [
                LinearParamScheduler(0.001, 1),  # warmup
                MultiStepParamScheduler(
                    [1, 0.1, 0.01, 0.001],
                    milestones=[10, 15, 20],
                    num_updates=30,
                ),
            ],
            interval_scaling=["rescaled", "fixed"],
            lengths=[5 / 30, 25 / 30],
        )
        sched = LRMultiplier(opt, multiplier, 30)
        # This is an equivalent of:
        # sched = WarmupMultiStepLR(
        # opt, milestones=[10, 15, 20], gamma=0.1, warmup_factor=0.001, warmup_iters=5)

        p.sum().backward()
        opt.step()

        lrs = [0.005]
        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        self.assertTrue(np.allclose(lrs[5:10], 5.0))
        self.assertTrue(np.allclose(lrs[10:15], 0.5))
        self.assertTrue(np.allclose(lrs[15:20], 0.05))
        self.assertTrue(np.allclose(lrs[20:], 0.005))

    def test_warmup_cosine(self):
        p = nn.Parameter(torch.zeros(0))
        opt = torch.optim.SGD([p], lr=5)
        multiplier = CompositeParamScheduler(
            [
                LinearParamScheduler(0.001, 1),  # warmup
                CosineParamScheduler(1, 0),
            ],
            interval_scaling=["rescaled", "fixed"],
            lengths=[5 / 30, 25 / 30],
        )
        sched = LRMultiplier(opt, multiplier, 30)

        p.sum().backward()
        opt.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.005)
        lrs = [0.005]

        for _ in range(30):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        for idx, lr in enumerate(lrs):
            expected_cosine = 2.5 * (1.0 + math.cos(math.pi * idx / 30))
            if idx >= 5:
                self.assertAlmostEqual(lr, expected_cosine)
            else:
                self.assertNotAlmostEqual(lr, expected_cosine)
