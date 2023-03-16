# Copyright (c) Facebook, Inc. and its affiliates.
from .config import (
    add_bootstrap_config,
    add_dataset_category_config,
    add_densepose_config,
    add_densepose_head_config,
    add_hrnet_config,
    load_bootstrap_config,
)
from .converters import builtin as builtin_converters  # register converters
from .data.datasets import builtin  # just to register data
from .evaluation import DensePoseCOCOEvaluator
from .modeling.hrfpn import build_hrfpn_backbone
from .modeling.roi_heads import DensePoseROIHeads
from .modeling.test_time_augmentation import (
    DensePoseDatasetMapperTTA,
    DensePoseGeneralizedRCNNWithTTA,
)
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from .utils.transform import load_from_cfg
