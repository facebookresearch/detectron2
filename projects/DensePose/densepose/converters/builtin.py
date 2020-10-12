# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from ..structures import DensePoseChartPredictorOutput
from . import ToMaskConverter, predictor_output_with_fine_and_coarse_segm_to_mask

ToMaskConverter.register(
    DensePoseChartPredictorOutput, predictor_output_with_fine_and_coarse_segm_to_mask
)
