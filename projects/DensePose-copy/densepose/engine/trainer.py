# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import time
import datetime
from collections import OrderedDict, abc
from typing import List, Optional, Union
import torch
from torch import nn
from contextlib import ExitStack

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    print_csv_format,
    inference_context
)
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.logger import log_every_n_seconds

from densepose import DensePoseDatasetMapperTTA, DensePoseGeneralizedRCNNWithTTA, load_from_cfg
from densepose.data import (
    DatasetMapper,
    build_combined_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    build_inference_based_loaders,
    has_inference_based_loaders,
)
from densepose.evaluation.d2_evaluator_adapter import Detectron2COCOEvaluatorAdapter
from densepose.evaluation.evaluator import DensePoseCOCOEvaluator, build_densepose_evaluator_storage
from densepose.modeling.cse import Embedder


class SampleCountingLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        it = iter(self.loader)
        storage = get_event_storage()
        while True:
            try:
                batch = next(it)
                num_inst_per_dataset = {}
                for data in batch:
                    dataset_name = data["dataset"]
                    if dataset_name not in num_inst_per_dataset:
                        num_inst_per_dataset[dataset_name] = 0
                    num_inst = len(data["instances"])
                    num_inst_per_dataset[dataset_name] += num_inst
                for dataset_name in num_inst_per_dataset:
                    storage.put_scalar(f"batch/{dataset_name}", num_inst_per_dataset[dataset_name])
                yield batch
            except StopIteration:
                break


class SampleCountMetricPrinter(EventWriter):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write(self):
        storage = get_event_storage()
        batch_stats_strs = []
        for key, buf in storage.histories().items():
            if key.startswith("batch/"):
                batch_stats_strs.append(f"{key} {buf.avg(20)}")
        self.logger.info(", ".join(batch_stats_strs))


class Trainer(DefaultTrainer):
    @classmethod
    def extract_embedder_from_model(cls, model: nn.Module) -> Optional[Embedder]:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "embedder"):
            return model.roi_heads.embedder
        return None

    # TODO: the only reason to copy the base class code here is to pass the embedder from
    # the model to the evaluator; that should be refactored to avoid unnecessary copy-pasting
    @classmethod
    def test(
        cls,
        cfg: CfgNode,
        model: nn.Module,
        evaluators: Optional[Union[DatasetEvaluator, List[DatasetEvaluator]]] = None,
    ):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (DatasetEvaluator, list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    embedder = cls.extract_embedder_from_model(model)
                    evaluator = cls.build_evaluator(cfg, dataset_name, embedder=embedder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE or comm.is_main_process():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            else:
                results_i = {}
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_evaluator(
        cls,
        cfg: CfgNode,
        dataset_name: str,
        output_folder: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        single: bool = False,
    ) -> DatasetEvaluators:
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = []
        distributed = cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE
        # Note: we currently use COCO evaluator for both COCO and LVIS datasets
        # to have compatible metrics. LVIS bbox evaluator could also be used
        # with an adapter to properly handle filtered / mapped categories
        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # if evaluator_type == "coco":
        #     evaluators.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # elif evaluator_type == "lvis":
        #     evaluators.append(LVISEvaluator(dataset_name, output_dir=output_folder))
        # evaluators.append(
        #     Detectron2COCOEvaluatorAdapter(
        #         dataset_name, output_dir=output_folder, distributed=distributed
        #     )
        # )
        if cfg.MODEL.DENSEPOSE_ON:
            storage = build_densepose_evaluator_storage(cfg, output_folder)
            if single:
                evaluators.append(
                    
                )
            else:
                evaluators.append(
                    DensePoseCOCOEvaluator(
                        dataset_name,
                        distributed,
                        output_folder,
                        evaluator_type=cfg.DENSEPOSE_EVALUATION.TYPE,
                        min_iou_threshold=cfg.DENSEPOSE_EVALUATION.MIN_IOU_THRESHOLD,
                        storage=storage,
                        embedder=embedder,
                        should_evaluate_mesh_alignment=cfg.DENSEPOSE_EVALUATION.EVALUATE_MESH_ALIGNMENT,
                        mesh_alignment_mesh_names=cfg.DENSEPOSE_EVALUATION.MESH_ALIGNMENT_MESH_NAMES,
                    )
                )
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "features": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.FEATURES_LR_FACTOR,
                },
                "embeddings": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_LR_FACTOR,
                },
            },
        )
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return maybe_add_gradient_clipping(cfg, optimizer)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        if not has_inference_based_loaders(cfg):
            return data_loader
        model = cls.build_model(cfg)
        model.to(cfg.BOOTSTRAP_MODEL.DEVICE)
        DetectionCheckpointer(model).resume_or_load(cfg.BOOTSTRAP_MODEL.WEIGHTS, resume=False)
        inference_based_loaders, ratios = build_inference_based_loaders(cfg, model)
        loaders = [data_loader] + inference_based_loaders
        ratios = [1.0] + ratios
        combined_data_loader = build_combined_loader(cfg, loaders, ratios)
        sample_counting_loader = SampleCountingLoader(combined_data_loader)
        return sample_counting_loader

    def build_writers(self):
        writers = super().build_writers()
        writers.append(SampleCountMetricPrinter())
        return writers

    @classmethod
    def test_with_TTA(cls, cfg: CfgNode, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        transform_data = load_from_cfg(cfg)
        model = DensePoseGeneralizedRCNNWithTTA(
            cfg, model, transform_data, DensePoseDatasetMapperTTA(cfg)
        )
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)  # pyre-ignore[6]
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
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
    num_devices = comm.get_world_size()
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

            # get gt boxes
            detected_persons = [x["instances"] for x in inputs]
            for person in detected_persons:
                person.pred_boxes = person.gt_boxes
                person.pred_classes = person.gt_classes
            
            outputs = model.inference(inputs, detected_persons, do_postprocess=False)
            # outputs = model(inputs)
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
