# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
from .data.datasets import builtin  # just to register data
from .converters import builtin as builtin_converters  # register converters
from .config import (
    add_densepose_config,
    add_densepose_head_config,
    add_hrnet_config,
    add_dataset_category_config,
    add_bootstrap_config,
    load_bootstrap_config,
)
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from .evaluation import DensePoseCOCOEvaluator
from .modeling.roi_heads import DensePoseROIHeads
from .modeling.test_time_augmentation import (
    DensePoseGeneralizedRCNNWithTTA,
    DensePoseDatasetMapperTTA,
)
from .utils.transform import load_from_cfg
from .modeling.hrfpn import build_hrfpn_backbone
