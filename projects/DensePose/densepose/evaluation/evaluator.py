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
from typing import Dict, Iterable, List, Optional
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO
from tabulate import tabulate

from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.comm import gather, get_rank, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from densepose.converters import ToChartResultConverter, ToMaskConverter
from densepose.data.datasets.coco import maybe_filter_and_map_categories_cocoapi
from densepose.structures import (
    DensePoseChartPredictorOutput,
    DensePoseEmbeddingPredictorOutput,
    quantize_densepose_chart_result,
)

from .densepose_coco_evaluation import DensePoseCocoEval, DensePoseEvalMode
from .mesh_alignment_evaluator import MeshAlignmentEvaluator
from .tensor_storage import (
    SingleProcessFileTensorStorage,
    SingleProcessRamTensorStorage,
    SingleProcessTensorStorage,
    SizeData,
    storage_gather,
)


class DensePoseCOCOEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir=None,
        evaluator_type: str = "iuv",
        min_iou_threshold: float = 0.5,
        storage: Optional[SingleProcessTensorStorage] = None,
        embedder=None,
        should_evaluate_mesh_alignment: bool = False,
        mesh_alignment_mesh_names: Optional[List[str]] = None,
    ):
        self._embedder = embedder
        self._distributed = distributed
        self._output_dir = output_dir
        self._evaluator_type = evaluator_type
        self._storage = storage
        self._should_evaluate_mesh_alignment = should_evaluate_mesh_alignment

        assert not (
            should_evaluate_mesh_alignment and embedder is None
        ), "Mesh alignment evaluation is activated, but no vertex embedder provided!"
        if should_evaluate_mesh_alignment:
            self._mesh_alignment_evaluator = MeshAlignmentEvaluator(
                embedder,
                mesh_alignment_mesh_names,
            )

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._min_threshold = min_iou_threshold
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        maybe_filter_and_map_categories_cocoapi(dataset_name, self._coco_api)

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
            prediction_list = prediction_to_dict(
                instances,
                input["image_id"],
                self._embedder,
                self._metadata.class_to_mesh_name,
                self._storage is not None,
            )
            if self._storage is not None:
                for prediction_dict in prediction_list:
                    dict_to_store = {}
                    for field_name in self._storage.data_schema:
                        dict_to_store[field_name] = prediction_dict[field_name]
                    record_id = self._storage.put(dict_to_store)
                    prediction_dict["record_id"] = record_id
                    prediction_dict["rank"] = get_rank()
                    for field_name in self._storage.data_schema:
                        del prediction_dict[field_name]
            self._predictions.extend(prediction_list)

    def evaluate(self, img_ids=None):
        if self._distributed:
            synchronize()
            predictions = gather(self._predictions)
            predictions = list(itertools.chain(*predictions))
        else:
            predictions = self._predictions

        multi_storage = storage_gather(self._storage) if self._storage is not None else None

        if not is_main_process():
            return
        return copy.deepcopy(self._eval_predictions(predictions, multi_storage, img_ids))

    def _eval_predictions(self, predictions, multi_storage=None, img_ids=None):
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
            self._coco_api,
            predictions,
            multi_storage,
            self._embedder,
            class_names=self._metadata.get("thing_classes"),
            min_threshold=self._min_threshold,
            img_ids=img_ids,
        )
        res["densepose_gps"] = results_gps
        res["densepose_gpsm"] = results_gpsm
        res["densepose_segm"] = results_segm
        if self._should_evaluate_mesh_alignment:
            res["densepose_mesh_alignment"] = self._evaluate_mesh_alignment()
        return res

    def _evaluate_mesh_alignment(self):
        self._logger.info("Mesh alignment evaluation ...")
        mean_ge, mean_gps, per_mesh_metrics = self._mesh_alignment_evaluator.evaluate()
        results = {
            "GE": mean_ge * 100,
            "GPS": mean_gps * 100,
        }
        mesh_names = set()
        for metric_name in per_mesh_metrics:
            for mesh_name, value in per_mesh_metrics[metric_name].items():
                results[f"{metric_name}-{mesh_name}"] = value * 100
                mesh_names.add(mesh_name)
        self._print_mesh_alignment_results(results, mesh_names)
        return results

    def _print_mesh_alignment_results(self, results: Dict[str, float], mesh_names: Iterable[str]):
        self._logger.info("Evaluation results for densepose, mesh alignment:")
        self._logger.info(f'| {"Mesh":13s} | {"GErr":7s} | {"GPS":7s} |')
        self._logger.info("| :-----------: | :-----: | :-----: |")
        for mesh_name in mesh_names:
            ge_key = f"GE-{mesh_name}"
            ge_str = f"{results[ge_key]:.4f}" if ge_key in results else " "
            gps_key = f"GPS-{mesh_name}"
            gps_str = f"{results[gps_key]:.4f}" if gps_key in results else " "
            self._logger.info(f"| {mesh_name:13s} | {ge_str:7s} | {gps_str:7s} |")
        self._logger.info("| :-------------------------------: |")
        ge_key = "GE"
        ge_str = f"{results[ge_key]:.4f}" if ge_key in results else " "
        gps_key = "GPS"
        gps_str = f"{results[gps_key]:.4f}" if gps_key in results else " "
        self._logger.info(f'| {"MEAN":13s} | {ge_str:7s} | {gps_str:7s} |')


def prediction_to_dict(instances, img_id, embedder, class_to_mesh_name, use_storage):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    raw_boxes_xywh = BoxMode.convert(
        instances.pred_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    )

    if isinstance(instances.pred_densepose, DensePoseEmbeddingPredictorOutput):
        results_densepose = densepose_cse_predictions_to_dict(
            instances, embedder, class_to_mesh_name, use_storage
        )
    elif isinstance(instances.pred_densepose, DensePoseChartPredictorOutput):
        if not use_storage:
            results_densepose = densepose_chart_predictions_to_dict(instances)
        else:
            results_densepose = densepose_chart_predictions_to_storage_dict(instances)

    results = []
    for k in range(len(instances)):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": raw_boxes_xywh[k].tolist(),
            "score": scores[k],
        }
        results.append({**result, **results_densepose[k]})
    return results


def densepose_chart_predictions_to_dict(instances):
    segmentations = ToMaskConverter.convert(
        instances.pred_densepose, instances.pred_boxes, instances.image_size
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
            "densepose": densepose_results_quantized,
            "segmentation": segmentation_encoded,
        }
        results.append(result)
    return results


def densepose_chart_predictions_to_storage_dict(instances):
    results = []
    for k in range(len(instances)):
        densepose_predictor_output = instances.pred_densepose[k]
        result = {
            "coarse_segm": densepose_predictor_output.coarse_segm.squeeze(0).cpu(),
            "fine_segm": densepose_predictor_output.fine_segm.squeeze(0).cpu(),
            "u": densepose_predictor_output.u.squeeze(0).cpu(),
            "v": densepose_predictor_output.v.squeeze(0).cpu(),
        }
        results.append(result)
    return results


def densepose_cse_predictions_to_dict(instances, embedder, class_to_mesh_name, use_storage):
    results = []
    for k in range(len(instances)):
        cse = instances.pred_densepose[k]
        results.append(
            {
                "coarse_segm": cse.coarse_segm[0].cpu(),
                "embedding": cse.embedding[0].cpu(),
            }
        )
    return results


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    multi_storage=None,
    embedder=None,
    class_names=None,
    min_threshold: float = 0.5,
    img_ids=None,
):
    logger = logging.getLogger(__name__)

    densepose_metrics = _get_densepose_metrics(min_threshold)
    if len(coco_results) == 0:  # cocoapi does not handle empty results very well
        logger.warn("No predictions from the model! Set scores to -1")
        results_gps = {metric: -1 for metric in densepose_metrics}
        results_gpsm = {metric: -1 for metric in densepose_metrics}
        results_segm = {metric: -1 for metric in densepose_metrics}
        return results_gps, results_gpsm, results_segm

    coco_dt = coco_gt.loadRes(coco_results)

    results = []
    for eval_mode_name in ["GPS", "GPSM", "IOU"]:
        eval_mode = getattr(DensePoseEvalMode, eval_mode_name)
        coco_eval = DensePoseCocoEval(
            coco_gt, coco_dt, "densepose", multi_storage, embedder, dpEvalMode=eval_mode
        )
        result = _derive_results_from_coco_eval(
            coco_eval, eval_mode_name, densepose_metrics, class_names, min_threshold, img_ids
        )
        results.append(result)
    return results


def _get_densepose_metrics(min_threshold: float = 0.5):
    metrics = ["AP"]
    if min_threshold <= 0.201:
        metrics += ["AP20"]
    if min_threshold <= 0.301:
        metrics += ["AP30"]
    if min_threshold <= 0.401:
        metrics += ["AP40"]
    metrics.extend(["AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"])
    return metrics


def _derive_results_from_coco_eval(
    coco_eval, eval_mode_name, metrics, class_names, min_threshold: float, img_ids
):
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.params.iouThrs = np.linspace(
        min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
    logger = logging.getLogger(__name__)
    logger.info(
        f"Evaluation results for densepose, {eval_mode_name} metric: \n"
        + create_small_table(results)
    )
    if class_names is None or len(class_names) <= 1:
        return results

    # Compute per-category AP, the same way as it is done in D2
    # (see detectron2/evaluation/coco_evaluation.py):
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append((f"{name}", float(ap * 100)))

    # tabulate it
    n_cols = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::n_cols] for i in range(n_cols)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (n_cols // 2),
        numalign="left",
    )
    logger.info(f"Per-category {eval_mode_name} AP: \n" + table)

    results.update({"AP-" + name: ap for name, ap in results_per_category})
    return results


def build_densepose_evaluator_storage(cfg: CfgNode, output_folder: str):
    storage_spec = cfg.DENSEPOSE_EVALUATION.STORAGE
    if storage_spec == "none":
        return None
    evaluator_type = cfg.DENSEPOSE_EVALUATION.TYPE
    # common output tensor sizes
    hout = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
    wout = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
    n_csc = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
    # specific output tensors
    if evaluator_type == "iuv":
        n_fsc = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        schema = {
            "coarse_segm": SizeData(dtype="float32", shape=(n_csc, hout, wout)),
            "fine_segm": SizeData(dtype="float32", shape=(n_fsc, hout, wout)),
            "u": SizeData(dtype="float32", shape=(n_fsc, hout, wout)),
            "v": SizeData(dtype="float32", shape=(n_fsc, hout, wout)),
        }
    elif evaluator_type == "cse":
        embed_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
        schema = {
            "coarse_segm": SizeData(dtype="float32", shape=(n_csc, hout, wout)),
            "embedding": SizeData(dtype="float32", shape=(embed_size, hout, wout)),
        }
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    # storage types
    if storage_spec == "ram":
        storage = SingleProcessRamTensorStorage(schema, io.BytesIO())
    elif storage_spec == "file":
        fpath = os.path.join(output_folder, f"DensePoseEvaluatorStorage.{get_rank()}.bin")
        PathManager.mkdirs(output_folder)
        storage = SingleProcessFileTensorStorage(schema, fpath, "wb")
    else:
        raise ValueError(f"Unknown storage specification: {storage_spec}")
    return storage
