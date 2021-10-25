#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import torch
from fvcore.nn.precise_bn import update_bn_stats

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation import inference_on_dataset
from detectron2.layers import CycleBatchNormList
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = setup_logger()
setup_logger(name="fvcore")


if __name__ == "__main__":
    checkpoint = sys.argv[1]
    cfg = LazyConfig.load_rel("./configs/retinanet_SyncBNhead.py")
    model = cfg.model
    model.head.norm = lambda c: CycleBatchNormList(len(model.head_in_features), num_features=c)
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
