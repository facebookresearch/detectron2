# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .densepose_uniform import DensePoseUniformSampler
from .densepose_confidence_based import DensePoseConfidenceBasedSampler
from .mask_from_densepose import MaskFromDensePoseSampler, densepose_to_mask
from .prediction_to_gt import PredictionToGroundTruthSampler
