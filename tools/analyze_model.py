# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
from collections import Counter
import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.analysis import (
    activation_count_operators,
    flop_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger()
    return cfg


def do_flop(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = flop_count_operators(model, data)
        counts += count
        total_flops.append(sum(count.values()))
    logger.info(
        "(G)Flops for Each Type of Operators:\n" + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info("Total (G)Flops: {}±{}".format(np.mean(total_flops), np.std(total_flops)))


def do_activation(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(cfg):
    model = build_model(cfg)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    model = build_model(cfg)
    logger.info("Model Structure:\n" + str(model))


if __name__ == "__main__":
    parser = default_argument_parser(
        epilog="""
Examples:

To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:

$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    cfg = setup(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
        }[task](cfg)
