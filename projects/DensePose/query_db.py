#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import os
import sys
from timeit import default_timer as timer
from typing import Any, ClassVar, Dict, List
import torch

from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from densepose.data.structures import DensePoseDataRelative
from densepose.utils.dbhelper import EntrySelector
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import BoundingBoxVisualizer
from densepose.vis.densepose_data_points import (
    DensePoseDataCoarseSegmentationVisualizer,
    DensePoseDataPointsIVisualizer,
    DensePoseDataPointsUVisualizer,
    DensePoseDataPointsVisualizer,
    DensePoseDataPointsVVisualizer,
)

DOC = """Query DB - a tool to print / visualize data from a database
"""

LOGGER_NAME = "query_db"

logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action(object):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class EntrywiseAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(EntrywiseAction, cls).add_arguments(parser)
        parser.add_argument(
            "dataset", metavar="<dataset>", help="Dataset name (e.g. densepose_coco_2014_train)"
        )
        parser.add_argument(
            "selector",
            metavar="<selector>",
            help="Dataset entry selector in the form field1[:type]=value1[,"
            "field2[:type]=value_min-value_max...] which selects all "
            "entries from the dataset that satisfy the constraints",
        )
        parser.add_argument(
            "--max-entries", metavar="N", help="Maximum number of entries to process", type=int
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        dataset = setup_dataset(args.dataset)
        entry_selector = EntrySelector.from_string(args.selector)
        context = cls.create_context(args)
        if args.max_entries is not None:
            for _, entry in zip(range(args.max_entries), dataset):
                if entry_selector(entry):
                    cls.execute_on_entry(entry, context)
        else:
            for entry in dataset:
                if entry_selector(entry):
                    cls.execute_on_entry(entry, context)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        context = {}
        return context


@register_action
class PrintAction(EntrywiseAction):
    """
    Print action that outputs selected entries to stdout
    """

    COMMAND: ClassVar[str] = "print"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Output selected entries to stdout. ")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(PrintAction, cls).add_arguments(parser)

    @classmethod
    def execute_on_entry(cls: type, entry: Dict[str, Any], context: Dict[str, Any]):
        import pprint

        printer = pprint.PrettyPrinter(indent=2, width=200, compact=True)
        printer.pprint(entry)


@register_action
class ShowAction(EntrywiseAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_segm": DensePoseDataCoarseSegmentationVisualizer(),
        "dp_i": DensePoseDataPointsIVisualizer(),
        "dp_u": DensePoseDataPointsUVisualizer(),
        "dp_v": DensePoseDataPointsVVisualizer(),
        "dp_pts": DensePoseDataPointsVisualizer(),
        "bbox": BoundingBoxVisualizer(),
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="output.png",
            help="File name to save output to",
        )

    @classmethod
    def execute_on_entry(cls: type, entry: Dict[str, Any], context: Dict[str, Any]):
        import cv2
        import numpy as np

        image_fpath = PathManager.get_local_path(entry["file_name"])
        image = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        datas = cls._extract_data_for_visualizers_from_entry(context["vis_specs"], entry)
        visualizer = context["visualizer"]
        image_vis = visualizer.visualize(image, datas)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        for vis_spec in vis_specs:
            vis = cls.VISUALIZERS[vis_spec]
            visualizers.append(vis)
        context = {
            "vis_specs": vis_specs,
            "visualizer": CompoundVisualizer(visualizers),
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context

    @classmethod
    def _extract_data_for_visualizers_from_entry(
        cls: type, vis_specs: List[str], entry: Dict[str, Any]
    ):
        dp_list = []
        bbox_list = []
        for annotation in entry["annotations"]:
            is_valid, _ = DensePoseDataRelative.validate_annotation(annotation)
            if not is_valid:
                continue
            bbox = torch.as_tensor(annotation["bbox"])
            bbox_list.append(bbox)
            dp_data = DensePoseDataRelative(annotation)
            dp_list.append(dp_data)
        datas = []
        for vis_spec in vis_specs:
            datas.append(bbox_list if "bbox" == vis_spec else (bbox_list, dp_list))
        return datas


def setup_dataset(dataset_name):
    logger.info("Loading dataset {}".format(dataset_name))
    start = timer()
    dataset = DatasetCatalog.get(dataset_name)
    stop = timer()
    logger.info("Loaded dataset {} in {:.3f}s".format(dataset_name, stop - start))
    return dataset


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    args.func(args)


if __name__ == "__main__":
    main()
