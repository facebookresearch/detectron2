#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import torch
from fvcore.nn.precise_bn import update_bn_stats
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = setup_logger()
setup_logger(name="fvcore")


class CycleBatchNormList(nn.ModuleList):
    """
    A hacky way to implement domain-specific BatchNorm
    if it's guaranteed that a fixed number of domains will be
    called with fixed order.
    """

    def __init__(self, length, channels):
        super().__init__([nn.BatchNorm2d(channels, affine=False) for k in range(length)])
        # shared affine, domain-specific BN
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self._pos = 0

    def forward(self, x):
        ret = self[self._pos](x)
        self._pos = (self._pos + 1) % len(self)

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        return ret * w + b


if __name__ == "__main__":
    checkpoint = sys.argv[1]
    cfg = LazyConfig.load_rel("./configs/retinanet_SyncBNhead.py")
    model = cfg.model
    model.head.norm = lambda c: CycleBatchNormList(len(model.head_in_features), c)
    model = instantiate(model)
    model.cuda()
    DetectionCheckpointer(model).load(checkpoint)

    cfg.dataloader.train.total_batch_size = 8
    logger.info("Running PreciseBN ...")
    with EventStorage(), torch.no_grad():
        update_bn_stats(model, instantiate(cfg.dataloader.train), 500)

    logger.info("Running evaluation ...")
    inference_on_dataset(
        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    )
