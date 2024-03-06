# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from .chart import DensePoseChartLoss
from .chart_with_confidences import DensePoseChartWithConfidenceLoss
from .cse import DensePoseCseLoss
from .registry import DENSEPOSE_LOSS_REGISTRY


__all__ = [
    "DensePoseChartLoss",
    "DensePoseChartWithConfidenceLoss",
    "DensePoseCseLoss",
    "DENSEPOSE_LOSS_REGISTRY",
]
