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
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2.data import (
    DatasetFromList,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.benchmark import DataLoaderBenchmark
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


def create_data_benchmark(cfg, args):
    if args.config_file.endswith(".py"):
        dl_cfg = cfg.dataloader.train
        dl_cfg._target_ = DataLoaderBenchmark
        return instantiate(dl_cfg)
    else:
        kwargs = build_detection_train_loader.from_config(cfg)
        kwargs.pop("aspect_ratio_grouping", None)
        kwargs["_target_"] = DataLoaderBenchmark
        return instantiate(kwargs)


def RAM_msg():
    vram = psutil.virtual_memory()
    return "RAM Usage: {:.2f}/{:.2f} GB".format(
        (vram.total - vram.available) / 1024 ** 3, vram.total / 1024 ** 3
    )


def benchmark_data(args):
    cfg = setup(args)
    logger.info("After spawning " + RAM_msg())

    benchmark = create_data_benchmark(cfg, args)
    benchmark.benchmark_distributed(250, 10)
    # test for a few more rounds
    for k in range(10):
        logger.info(f"Iteration {k} " + RAM_msg())
        benchmark.benchmark_distributed(250, 1)


def benchmark_data_advanced(args):
    # benchmark dataloader with more details to help analyze performance bottleneck
    cfg = setup(args)
    benchmark = create_data_benchmark(cfg, args)

    if comm.get_rank() == 0:
        benchmark.benchmark_dataset(100)
        benchmark.benchmark_mapper(100)
        benchmark.benchmark_workers(100, warmup=10)
        benchmark.benchmark_IPC(100, warmup=10)
    if comm.get_world_size() > 1:
        benchmark.benchmark_distributed(100)
        logger.info("Rerun ...")
        benchmark.benchmark_distributed(100)


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
        [
            hooks.IterationTimer(),
            hooks.PeriodicWriter([CommonMetricPrinter(max_iter)]),
            hooks.TorchProfiler(
                lambda trainer: trainer.iter == max_iter - 1, cfg.OUTPUT_DIR, save_tensorboard=True
            ),
        ]
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
    parser.add_argument("--task", choices=["train", "eval", "data", "data_advanced"], required=True)
    args = parser.parse_args()
    assert not args.eval_only

    logger.info("Environment info:\n" + collect_env_info())
    if "data" in args.task:
        print("Initial " + RAM_msg())
    if args.task == "data":
        f = benchmark_data
    if args.task == "data_advanced":
        f = benchmark_data_advanced
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
