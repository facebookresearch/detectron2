# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from .converters import ToChartResultConverter, ToMaskConverter
from .densepose_coco_evaluation import DensePoseCocoEval, DensePoseEvalMode
from .structures import quantize_densepose_chart_result


class DensePoseCOCOEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._min_threshold = 0.5
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
            if not instances.has("pred_densepose"):
                continue
            self._predictions.extend(prediction_to_dict(instances, input["image_id"]))

    def evaluate(self, img_ids=None):
        if self._distributed:
            synchronize()
            predictions = all_gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
            if not is_main_process():
                return
        else:
            predictions = self._predictions

        return copy.deepcopy(self._eval_predictions(predictions, img_ids))

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions on densepose.
        Return results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "coco_densepose_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._logger.info("Evaluating predictions ...")
        res = OrderedDict()
        results_gps, results_gpsm, results_segm = _evaluate_predictions_on_coco(
            self._coco_api, predictions, min_threshold=self._min_threshold, img_ids=img_ids
        )
        res["densepose_gps"] = results_gps
        res["densepose_gpsm"] = results_gpsm
        res["densepose_segm"] = results_segm
        return res


def prediction_to_dict(instances, img_id):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    scores = instances.scores.tolist()
    segmentations = ToMaskConverter.convert(
        instances.pred_densepose, instances.pred_boxes, instances.image_size
    )
    raw_boxes_xywh = BoxMode.convert(
        instances.pred_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    )

    results = []
    for k in range(len(instances)):
        densepose_results_quantized = quantize_densepose_chart_result(
            ToChartResultConverter.convert(instances.pred_densepose[k], instances.pred_boxes[k])
        )
        densepose_results_quantized.labels_uv_uint8 = (
            densepose_results_quantized.labels_uv_uint8.cpu()
        )
        segmentation = segmentations.tensor[k]
        segmentation_encoded = mask_utils.encode(
            np.require(segmentation.numpy(), dtype=np.uint8, requirements=["F"])
        )
        segmentation_encoded["counts"] = segmentation_encoded["counts"].decode("utf-8")
        result = {
            "image_id": img_id,
            "category_id": 1,  # densepose only has one class
            "bbox": raw_boxes_xywh[k].tolist(),
            "score": scores[k],
            "densepose": densepose_results_quantized,
            "segmentation": segmentation_encoded,
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, min_threshold=0.5, img_ids=None):
    logger = logging.getLogger(__name__)

    segm_metrics = _get_segmentation_metrics()
    densepose_metrics = _get_densepose_metrics(min_threshold)
    if len(coco_results) == 0:  # cocoapi does not handle empty results very well
        logger.warn("No predictions from the model! Set scores to -1")
        results_gps = {metric: -1 for metric in densepose_metrics}
        results_gpsm = {metric: -1 for metric in densepose_metrics}
        results_segm = {metric: -1 for metric in segm_metrics}
        return results_gps, results_gpsm, results_segm

    coco_dt = coco_gt.loadRes(coco_results)
    results_segm = _evaluate_predictions_on_coco_segm(
        coco_gt, coco_dt, segm_metrics, min_threshold, img_ids
    )
    logger.info("Evaluation results for densepose segm: \n" + create_small_table(results_segm))
    results_gps = _evaluate_predictions_on_coco_gps(
        coco_gt, coco_dt, densepose_metrics, min_threshold, img_ids
    )
    logger.info(
        "Evaluation results for densepose, GPS metric: \n" + create_small_table(results_gps)
    )
    results_gpsm = _evaluate_predictions_on_coco_gpsm(
        coco_gt, coco_dt, densepose_metrics, min_threshold, img_ids
    )
    logger.info(
        "Evaluation results for densepose, GPSm metric: \n" + create_small_table(results_gpsm)
    )
    return results_gps, results_gpsm, results_segm


def _get_densepose_metrics(min_threshold=0.5):
    metrics = ["AP"]
    if min_threshold <= 0.201:
        metrics += ["AP20"]
    if min_threshold <= 0.301:
        metrics += ["AP30"]
    if min_threshold <= 0.401:
        metrics += ["AP40"]
    metrics.extend(["AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"])
    return metrics


def _get_segmentation_metrics():
    return [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR@1",
        "AR@10",
        "AR@100",
        "ARs",
        "ARm",
        "ARl",
    ]


def _evaluate_predictions_on_coco_gps(coco_gt, coco_dt, metrics, min_threshold=0.5, img_ids=None):
    coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "densepose", dpEvalMode=DensePoseEvalMode.GPS)
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.linspace(
        min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    return results


def _evaluate_predictions_on_coco_gpsm(coco_gt, coco_dt, metrics, min_threshold=0.5, img_ids=None):
    coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "densepose", dpEvalMode=DensePoseEvalMode.GPSM)
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.linspace(
        min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    return results


def _evaluate_predictions_on_coco_segm(coco_gt, coco_dt, metrics, min_threshold=0.5, img_ids=None):
    coco_eval = DensePoseCocoEval(coco_gt, coco_dt, "segm")
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.linspace(
        min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    return results
