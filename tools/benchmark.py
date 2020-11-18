#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script to benchmark builtin models.

Note: this script has an extra dependency of psutil.
"""

import itertools
import logging
import psutil
import torch
import tqdm
from fvcore.common.timer import Timer
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetFromList,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


def RAM_msg():
    vram = psutil.virtual_memory()
    return "RAM Usage: {:.2f}/{:.2f} GB".format(
        (vram.total - vram.available) / 1024 ** 3, vram.total / 1024 ** 3
    )


def benchmark_data(args):
    cfg = setup(args)

    logger.info("After spawning " + RAM_msg())
    timer = Timer()
    dataloader = build_detection_train_loader(cfg)
    logger.info("Initialize loader using {} seconds.".format(timer.seconds()))

    timer.reset()
    itr = iter(dataloader)
    for i in range(10):  # warmup
        next(itr)
        if i == 0:
            startup_time = timer.seconds()
    logger.info("Startup time: {} seconds".format(startup_time))
    timer = Timer()
    max_iter = 1000
    for _ in tqdm.trange(max_iter):
        next(itr)
    logger.info(
        "{} iters ({} images) in {} seconds.".format(
            max_iter, max_iter * cfg.SOLVER.IMS_PER_BATCH, timer.seconds()
        )
    )

    # test for a few more rounds
    for k in range(10):
        logger.info(f"Iteration {k} " + RAM_msg())
        timer = Timer()
        max_iter = 1000
        for _ in tqdm.trange(max_iter):
            next(itr)
        logger.info(
            "{} iters ({} images) in {} seconds.".format(
                max_iter, max_iter * cfg.SOLVER.IMS_PER_BATCH, timer.seconds()
            )
        )


def benchmark_train(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    optimizer = build_optimizer(cfg, model)
    checkpointer = DetectionCheckpointer(model, optimizer=optimizer)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 2
    data_loader = build_detection_train_loader(cfg)
    dummy_data = list(itertools.islice(data_loader, 100))

    def f():
        data = DatasetFromList(dummy_data, copy=False, serialize=False)
        while True:
            yield from data

    max_iter = 400
    trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, f(), optimizer)
    trainer.register_hooks(
        [hooks.IterationTimer(), hooks.PeriodicWriter([CommonMetricPrinter(max_iter)])]
    )
    trainer.train(1, max_iter)


@torch.no_grad()
def benchmark_eval(args):
    cfg = setup(args)
    model = build_model(cfg)
    model.eval()
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    dummy_data = DatasetFromList(list(itertools.islice(data_loader, 100)), copy=False)

    def f():
        while True:
            yield from dummy_data

    for k in range(5):  # warmup
        model(dummy_data[k])

    max_iter = 300
    timer = Timer()
    with tqdm.tqdm(total=max_iter) as pbar:
        for idx, d in enumerate(f()):
            if idx == max_iter:
                break
            model(d)
            pbar.update()
    logger.info("{} iters in {} seconds.".format(max_iter, timer.seconds()))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only

    if args.task == "data":
        f = benchmark_data
        print("Initial " + RAM_msg())
    elif args.task == "train":
        """
        Note: training speed may not be representative.
        The training cost of a R-CNN model varies with the content of the data
        and the quality of the model.
        """
        f = benchmark_train
    elif args.task == "eval":
        f = benchmark_eval
        # only benchmark single-GPU inference.
        assert args.num_gpus == 1 and args.num_machines == 1
    launch(f, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))
