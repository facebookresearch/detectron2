# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from .densepose_uniform import DensePoseUniformSampler
from .densepose_confidence_based import DensePoseConfidenceBasedSampler
from .densepose_cse_uniform import DensePoseCSEUniformSampler
from .densepose_cse_confidence_based import DensePoseCSEConfidenceBasedSampler
from .mask_from_densepose import MaskFromDensePoseSampler
from .prediction_to_gt import PredictionToGroundTruthSampler
