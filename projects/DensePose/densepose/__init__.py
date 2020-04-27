# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .data.datasets import builtin  # just to register data
from .config import add_densepose_config, add_dataset_category_config
from .densepose_head import ROI_DENSEPOSE_HEAD_REGISTRY
from .evaluator import DensePoseCOCOEvaluator
from .roi_head import DensePoseROIHeads
from .data.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from .modeling.test_time_augmentation import DensePoseGeneralizedRCNNWithTTA
from .utils.transform import load_from_cfg
