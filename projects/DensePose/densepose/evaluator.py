# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.logger import create_small_table

from .densepose_coco_evaluation import DensePoseCocoEval, DensePoseEvalMode


class DensePoseCOCOEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
                The :class:`Instances` object needs to have `densepose` field.
        """
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to(self._cpu_device)

            boxes = instances.pred_boxes.tensor.clone()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            instances.pred_densepose = instances.pred_densepose.to_result(boxes)

            json_results = prediction_to_json(instances, input["image_id"])
            self._predictions.extend(json_results)

    def evaluate(self):
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            if not is_main_process():
                return
        else:
            predictions = self._predictions

        return copy.deepcopy(self._eval_predictions(predictions))

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions on densepose.
        Return results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_densepose_results.json")
            with open(file_path, "w") as f:
                json.dump(predictions, f)
                f.flush()
                os.fsync(f.fileno())

        self._logger.info("Evaluating predictions ...")
        res = OrderedDict()
        results_gps, results_gpsm = _evaluate_predictions_on_coco(self._coco_api, predictions)
        res["densepose_gps"] = results_gps
        res["densepose_gpsm"] = results_gpsm
        return res


def prediction_to_json(instances, img_id):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    scores = instances.scores.tolist()

    results = []
    for k in range(len(instances)):
        densepose = instances.pred_densepose[k]
        result = {
            "image_id": img_id,
            "category_id": 1,  # densepose only has one class
            "bbox": densepose[1],
            "score": scores[k],
            "densepose": densepose,
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results):
    metrics = ["AP", "AP50", "AP75", "APm", "APl"]

    logger = logging.getLogger(__name__)

    if len(coco_results) == 0:  # cocoapi does not handle empty results very well
        logger.warn("No predictions from the model! Set scores to -1")
        results_gps = {metric: -1 for metric in metrics}
        results_gpsm = {metric: -1 for metric in metrics}
        return results_gps, results_gpsm

    coco_dt = coco_gt.loadRes(coco_results)
    results_gps = _evaluate_predictions_on_coco_gps(coco_gt, coco_dt, metrics)
    logger.info(
        "Evaluation results for densepose, GPS metric: \n" + create_small_table(results_gps)
    )
    results_gpsm = _evaluate_predictions_on_coco_gpsm(coco_gt, coco_dt, metrics)
    logger.info(
        "Evaluation results for densepose, GPSm metric: \n" + create_small_table(results_gpsm)
    )
    return results_gps, results_gpsm


def _evaluate_predictions_on_coco_gps(coco_gt, coco_dt, metrics):
    coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "densepose", dpEvalMode=DensePoseEvalMode.GPS)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    return results


def _evaluate_predictions_on_coco_gpsm(coco_gt, coco_dt, metrics):
    coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "densepose", dpEvalMode=DensePoseEvalMode.GPSM)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    return results
