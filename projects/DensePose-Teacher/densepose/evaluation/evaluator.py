# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import logging
import numpy as np
import os
from collections import OrderedDict, abc
from typing import Dict, Iterable, List, Optional, Union
import pycocotools.mask as mask_utils
import torch
from torch import nn
from pycocotools.coco import COCO
from tabulate import tabulate
from contextlib import ExitStack
import time
import datetime

from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators , inference_context
from detectron2.structures import BoxMode
from detectron2.utils.comm import gather, get_rank, is_main_process, synchronize, get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from detectron2.modeling.meta_arch import GeneralizedRCNN

from densepose.converters import ToChartResultConverter, ToMaskConverter, ToChartResultConverterWithBlock
from densepose.data.datasets.coco import maybe_filter_and_map_categories_cocoapi
from densepose.structures import (
    DensePoseChartPredictorOutput,
    DensePoseEmbeddingPredictorOutput,
    quantize_densepose_chart_result,
    BlockPredictorOutput,
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

from densepose.modeling.correction import Accumulator


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
        block_num=5,
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

        self.block_num = block_num

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
                self.block_num,
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


def prediction_to_dict(instances, img_id, embedder, class_to_mesh_name, use_storage, block_num):
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
    elif isinstance(instances.pred_densepose, BlockPredictorOutput):
        results_densepose = densepose_block_predictions_to_dict(instances, block_num)

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


def densepose_block_predictions_to_dict(instances, block_num):
    segmentations = ToMaskConverter.convert(
        instances.pred_densepose, instances.pred_boxes, instances.image_size
    )

    results = []
    for k in range(len(instances)):
        chart_result = ToChartResultConverterWithBlock.convert(
            instances.pred_densepose[k], instances.pred_boxes[k], block_num=block_num
        )
        densepose_results_quantized = quantize_densepose_chart_result(chart_result)
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
    min_threshold: float=0.5,
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


def _get_densepose_metrics(min_threshold: float=0.5):
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


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], corrector=None
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            if corrector is not None:
                outputs = model.inference(inputs, do_postprocess=False)
                correction, _ = corrector(outputs)
                corrector.correct(correction, outputs)
                outputs = GeneralizedRCNN._postprocess(outputs, inputs, model.preprocess_image(inputs))
            else:
                outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_single_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], corrector=None
):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()

            detected_persons = [x["instances"] for x in inputs]
            for person in detected_persons:
                person.pred_boxes = person.gt_boxes
                person.pred_classes = person.gt_classes

            outputs = model.inference(inputs, detected_persons, do_postprocess=False)
            if corrector is not None:
                # outputs = model.inference(inputs, do_postprocess=False)
                correction, _ = corrector(outputs)
                corrector.correct(correction, outputs)
                # outputs = GeneralizedRCNN._postprocess(outputs, inputs, model.preprocess_image(inputs))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


class DensePoseCOCOSingleEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir=None,
        evaluator_type: str = "iuv",
        min_iou_threshold: float = 0.5,
        storage: Optional[SingleProcessTensorStorage] = None,
    ):
        # self._embedder = embedder
        self._distributed = distributed
        self._output_dir = output_dir
        self._evaluator_type = evaluator_type
        self._storage = storage

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._min_threshold = min_iou_threshold

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
            # instances = output["instances"].to(self._cpu_device)
            instances = output.to(self._cpu_device)
            if not instances.has("pred_densepose"):
                continue
            prediction_list = prediction_single_to_dict(
                instances,
                input["image_id"],
                self._metadata.class_to_mesh_name,
                self._storage is not None
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

        dists_results, i_compare, u_compare, v_compare = _evaluate_predictions_single_on_coco(predictions)

        return res

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


def prediction_single_to_dict(instances, img_id, class_to_mesh_name, use_storage):
    classes = instances.pred_classes.tolist()
    raw_boxes_xywh = BoxMode.convert(
        instances.pred_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    )

    # gt_densepose = densepose_gt_to_dict(instances)

    # if isinstance(instances.pred_densepose, DensePoseEmbeddingPredictorOutput):
    #     results_densepose = densepose_cse_predictions_to_dict(
    #         instances, embedder, class_to_mesh_name, use_storage
    #     )
    # elif isinstance(instances.pred_densepose, DensePoseChartPredictorOutput):
    #     if not use_storage:
    #         results_densepose = densepose_chart_predictions_to_dict(instances) # list of dict
    #     else:
    #         results_densepose = densepose_chart_predictions_to_storage_dict(instances)
    accumulator = Accumulator()
    accumulator.accumulate(instances)
    packed_annotations = accumulator.pack()
    results_densepose = {
        "packed_annotations": packed_annotations,
        "pred_densepose": {
            "u": instances.pred_densepose.u,
            "v": instances.pred_densepose.v,
            "fine_segm": instances.pred_densepose.fine_segm,
        }
    }

    results = []
    # for k in range(len(instances)):
    #     result = {
    #         "image_id": img_id,
    #         "category_id": classes[k],
    #         "bbox": raw_boxes_xywh[k].tolist(),
    #         "gt_densepose": gt_densepose[k],
    #     }
    #     results.append({**result, **results_densepose[k]})
    result = {
        "image_id": img_id,
        "category_id": classes,
        "bbox": raw_boxes_xywh,
    }
    results.append({**result, **results_densepose})
    return results


def _evaluate_predictions_single_on_coco(predictions):
    from scipy.io import loadmat
    import scipy.spatial.distance as ssd
    import pickle
    from densepose.modeling.correction import InterpolationHelper

    smpl_subdiv_fpath = PathManager.get_local_path(
        "https://dl.fbaipublicfiles.com/densepose/data/SMPL_subdiv.mat"
    )
    pdist_transform_fpath = PathManager.get_local_path(
        "https://dl.fbaipublicfiles.com/densepose/data/SMPL_SUBDIV_TRANSFORM.mat"
    )
    pdist_matrix_fpath = PathManager.get_local_path(
        "https://dl.fbaipublicfiles.com/densepose/data/Pdist_matrix.pkl", timeout_sec=120
    )
    SMPL_subdiv = loadmat(smpl_subdiv_fpath)
    PDIST_transform = loadmat(pdist_transform_fpath)
    PDIST_transform = PDIST_transform["index"].squeeze()
    UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
    ClosestVertInds = np.arange(UV.shape[1]) + 1
    Part_UVs = []
    Part_ClosestVertInds = []
    for i in np.arange(24):
        Part_UVs.append(UV[:, SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)])
        Part_ClosestVertInds.append(
            ClosestVertInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]
        )

    with open(pdist_matrix_fpath, "rb") as hFile:
        arrays = pickle.load(hFile, encoding="latin1")
    Pdist_matrix = arrays["Pdist_matrix"]

    dists_results = []

    i_compare = []
    u_compare = []
    v_compare = []

    n = 27554
    for prediction in predictions:
        # gt_densepose = prediction['gt_densepose']
        packed_annotations = prediction['packed_annotations']
        if packed_annotations is None:
            continue
        _, _, w_gt, h_gt = prediction['bbox']
        dp_x = (gt_densepose.x * w_gt / 256.0).int()
        dp_y = (gt_densepose.y * h_gt / 256.0).int()
        dp_x = dp_x[dp_x < w_gt]

        densepose_results_quantized = prediction["densepose"]
        i_est, u_est, v_est = _extract_iuv(
            densepose_results_quantized.labels_uv_uint8.numpy(), dp_y, dp_x
        )

        # pred_densepose = prediction['pred_densepose']
        # packed_annotations = prediction['packed_annotations']
        # h, w = pred_densepose['u'].shape[2:]
        # interpolator = InterpolationHelper.from_matches(packed_annotations, (h ,w))
        # u_est = interpolator.extract_at_points(pred_densepose['u'])
        # v_est = interpolator.extract_at_points(pred_densepose['v'])
        # i_est = interpolator.extract_at_points(
        #     pred_densepose['fine_segm'],
        #     slice_fine_segm=slice(None),
        #     w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
        #     w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
        #     w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
        #     w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        # ).argmax(dim=1).long()
        # i_gt, u_gt, v_gt = packed_annotations.fine_segm_labels_gt, packed_annotations.u_gt, packed_annotations.v_gt

        # get closest verts gt
        closest_vertsGT = np.ones(i_gt.shape) * -1
        closest_verts = np.ones(i_gt.shape) * -1
        for i in np.arange(24):
            current_part_uvs = Part_UVs[i]
            current_part_closest_vert_inds = Part_ClosestVertInds[i]
            if (i + 1) in i_gt:
                uvs = np.array([u_gt[i_gt == (i + 1)].numpy(), v_gt[i_gt == (i + 1)].numpy()])
                d = ssd.cdist(current_part_uvs.transpose(), uvs.transpose()).squeeze()
                closest_vertsGT[i_gt == (i + 1)] = current_part_closest_vert_inds[np.argmin(d, axis=0)]
            if (i + 1) in i_est:
                uvs = np.stack([u_est[i_est == (i + 1)], v_est[i_est == (i + 1)]])
                d = ssd.cdist(current_part_uvs.transpose(), uvs.transpose()).squeeze()
                closest_verts[i_est == (i + 1)] = current_part_closest_vert_inds[np.argmin(d, axis=0)]
        closest_verts_gt_transformed = PDIST_transform[closest_vertsGT.astype(int) - 1]
        closest_verts_gt_transformed[closest_vertsGT < 0] = 0
        closest_verts_transformed = PDIST_transform[closest_verts.astype(int) - 1]
        closest_verts_transformed[closest_verts < 0] = 0

        i_compare.append(np.stack((i_gt.numpy(), i_est), axis=1))
        u_compare.append(np.stack((u_gt.numpy(), u_est), axis=1))
        v_compare.append(np.stack((v_gt.numpy(), v_est), axis=1))

        dists = []
        for dd in range(len(closest_verts_gt_transformed)):
            if closest_verts_gt_transformed[dd] > 0:
                if closest_verts_transformed[dd] > 0:
                    i = closest_verts_gt_transformed[dd] - 1
                    j = closest_verts_transformed[dd] - 1
                    if j == i:
                        dists.append(0)
                    elif j > i:
                        ccc = i
                        i = j
                        j = ccc
                        i = n - i - 1
                        j = n - j - 1
                        k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                        k = (n * n - n) / 2 - k - 1
                        dists.append(Pdist_matrix[int(k)][0])
                    else:
                        i = n - i - 1
                        j = n - j - 1
                        k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                        k = (n * n - n) / 2 - k - 1
                        dists.append(Pdist_matrix[int(k)][0])
                else:
                    dists.append(np.inf)
        dists_results.append(np.array(dists))
    return np.concatenate(dists_results), np.concatenate(i_compare), np.concatenate(u_compare), np.concatenate(v_compare)


def _extract_iuv(densepose_data, dp_y, dp_x):
    return densepose_data[0, dp_y, dp_x], densepose_data[1, dp_y, dp_x] / 255.0, densepose_data[2, dp_y, dp_x] / 255.0


def densepose_gt_to_dict(instances):
    results = []
    gt_densepose = instances.gt_densepose
    for dp_gt in gt_densepose:
        if dp_gt is None or len(dp_gt.x) <= 0:
            result = None
        else:
            result = dp_gt
        
        results.append(result)
    return results
