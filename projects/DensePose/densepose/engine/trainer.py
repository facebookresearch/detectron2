# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.utils.events import EventWriter, get_event_storage

from densepose import (
    DensePoseCOCOEvaluator,
    DensePoseDatasetMapperTTA,
    DensePoseGeneralizedRCNNWithTTA,
    load_from_cfg,
)
from densepose.data import (
    DatasetMapper,
    build_combined_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    build_inference_based_loaders,
    has_inference_based_loaders,
)


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
    def build_evaluator(cls, cfg: CfgNode, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        if cfg.MODEL.DENSEPOSE_ON:
            evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
        return DatasetEvaluators(evaluators)

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
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
