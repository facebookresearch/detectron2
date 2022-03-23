# Copyright (c) Facebook, Inc. and its affiliates.

from .chart import DensePoseChartLoss
from .chart_with_confidences import DensePoseChartWithConfidenceLoss
from .cse import DensePoseCseLoss
from .score import ScoringLoss, IoULoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .gps import DenseScoreLossHelper


__all__ = [
    "DensePoseChartLoss",
    "DensePoseChartWithConfidenceLoss",
    "DensePoseCseLoss",
    "ScoringLoss",
    "IoULoss"
    "DenseScoreLossHelper",
    "DENSEPOSE_LOSS_REGISTRY",
]
