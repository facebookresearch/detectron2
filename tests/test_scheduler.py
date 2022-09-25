# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np
from unittest import TestCase
import torch
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler,
    StepWithFixedGammaParamScheduler,
)
from torch import nn

from detectron2.solver import LRMultiplier, WarmupParamScheduler, build_lr_scheduler


class TestScheduler(TestCase):
    def test_warmup_multistep(self):
        p = nn.Parameter(torch.zeros(0))
        opt = torch.optim.SGD([p], lr=5)

        multiplier = WarmupParamScheduler(
            MultiStepParamScheduler(
                [1, 0.1, 0.01, 0.001],
                milestones=[10, 15, 20],
                num_updates=30,
            ),
            0.001,
            5 / 30,
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
        multiplier = WarmupParamScheduler(
            CosineParamScheduler(1, 0),
            0.001,
            5 / 30,
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

    def test_warmup_cosine_end_value(self):
        from detectron2.config import CfgNode, get_cfg

        def _test_end_value(cfg_dict):
            cfg = get_cfg()
            cfg.merge_from_other_cfg(CfgNode(cfg_dict))

            p = nn.Parameter(torch.zeros(0))
            opt = torch.optim.SGD([p], lr=cfg.SOLVER.BASE_LR)

            scheduler = build_lr_scheduler(cfg, opt)

            p.sum().backward()
            opt.step()
            self.assertEqual(
                opt.param_groups[0]["lr"], cfg.SOLVER.BASE_LR * cfg.SOLVER.WARMUP_FACTOR
            )

            lrs = []
            for _ in range(cfg.SOLVER.MAX_ITER):
                scheduler.step()
                lrs.append(opt.param_groups[0]["lr"])

            self.assertAlmostEqual(lrs[-1], cfg.SOLVER.BASE_LR_END)

        _test_end_value(
            {
                "SOLVER": {
                    "LR_SCHEDULER_NAME": "WarmupCosineLR",
                    "MAX_ITER": 100,
                    "WARMUP_ITERS": 10,
                    "WARMUP_FACTOR": 0.1,
                    "BASE_LR": 5.0,
                    "BASE_LR_END": 0.0,
                }
            }
        )

        _test_end_value(
            {
                "SOLVER": {
                    "LR_SCHEDULER_NAME": "WarmupCosineLR",
                    "MAX_ITER": 100,
                    "WARMUP_ITERS": 10,
                    "WARMUP_FACTOR": 0.1,
                    "BASE_LR": 5.0,
                    "BASE_LR_END": 0.5,
                }
            }
        )

    def test_warmup_stepwithfixedgamma(self):
        p = nn.Parameter(torch.zeros(0))
        opt = torch.optim.SGD([p], lr=5)

        multiplier = WarmupParamScheduler(
            StepWithFixedGammaParamScheduler(
                base_value=1.0,
                gamma=0.1,
                num_decays=4,
                num_updates=30,
            ),
            0.001,
            5 / 30,
            rescale_interval=True,
        )
        sched = LRMultiplier(opt, multiplier, 30)

        p.sum().backward()
        opt.step()

        lrs = [0.005]
        for _ in range(29):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertTrue(np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001]))
        self.assertTrue(np.allclose(lrs[5:10], 5.0))
        self.assertTrue(np.allclose(lrs[10:15], 0.5))
        self.assertTrue(np.allclose(lrs[15:20], 0.05))
        self.assertTrue(np.allclose(lrs[20:25], 0.005))
        self.assertTrue(np.allclose(lrs[25:], 0.0005))

        # Calling sche.step() after the last training iteration is done will trigger IndexError
        with self.assertRaises(IndexError, msg="list index out of range"):
            sched.step()
