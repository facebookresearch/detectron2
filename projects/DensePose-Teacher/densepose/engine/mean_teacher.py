# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from bisect import bisect_right
import logging
from collections import Counter
import torch

from detectron2.engine import HookBase

class MeanTeacher(HookBase):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert 0 <= momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_train(self):
        self.momentum_update(self.trainer.teacher_model, self.trainer.student_model, 0)

    def before_step(self):
        if (self.trainer.iter + 1) % self.interval == 0:
            momentum = min(
                self.momentum, 1 - (1 + self.warm_up) / (self.trainer.iter + 1 + self.warm_up)
            )
            self.momentum_update(self.trainer.teacher_model, self.trainer.student_model, momentum)
        # pass

    def after_step(self):
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, self.trainer.iter
        )
        # pass

    def momentum_update(self, teacher_model, student_model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            student_model.named_parameters(), teacher_model.named_parameters()
        ):
            # if 'corrector' in src_name:
            #     continue
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

    def state_dict(self):
        return {"meanteacher_momentum": self.momentum}

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        logger.info("Loading mean teacher from state_dict ...")
        self.momentum = state_dict["meanteacher_momentum"]
