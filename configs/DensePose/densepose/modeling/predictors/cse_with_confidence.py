# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from . import DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingPredictor
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseEmbeddingWithConfidencePredictor(
    DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingPredictor
):
    """
    Predictor that combines CSE and CSE confidence estimation
    """

    pass
