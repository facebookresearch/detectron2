# Copyright (c) Facebook, Inc. and its affiliates.

from .chart import DensePoseChartPredictor
from .chart_confidence import DensePoseChartConfidencePredictorMixin
from .chart_with_confidence import DensePoseChartWithConfidencePredictor
from .cse import DensePoseEmbeddingPredictor
from .cse_confidence import DensePoseEmbeddingConfidencePredictorMixin
from .cse_with_confidence import DensePoseEmbeddingWithConfidencePredictor
from .registry import DENSEPOSE_PREDICTOR_REGISTRY
