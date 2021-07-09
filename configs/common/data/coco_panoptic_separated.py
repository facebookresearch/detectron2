from detectron2.config import LazyCall as L
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)

from .coco import dataloader

dataloader.train.dataset.names = "coco_2017_train_panoptic_separated"
dataloader.train.dataset.filter_empty = False
dataloader.test.dataset.names = "coco_2017_val_panoptic_separated"


dataloader.evaluator = [
    L(COCOEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(SemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
]
