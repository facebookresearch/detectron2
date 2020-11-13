# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_pointrend_config
from .mask_head import PointRendMaskHead
from .semantic_seg import PointRendSemSegHead
from .color_augmentation import ColorAugSSDTransform

from . import roi_heads as _  # only registration
