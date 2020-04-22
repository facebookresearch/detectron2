# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .coco import DATASETS as COCO_DATASETS
from .coco import register_datasets as register_coco_datasets

DEFAULT_DATASETS_ROOT = "datasets"


register_coco_datasets(COCO_DATASETS, DEFAULT_DATASETS_ROOT)
