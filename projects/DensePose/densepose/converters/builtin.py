# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from ..structures import DensePoseChartPredictorOutput
from . import (
    HFlipConverter,
    ToChartResultConverter,
    ToMaskConverter,
    densepose_chart_predictor_output_hflip,
    densepose_chart_predictor_output_to_result,
    predictor_output_with_fine_and_coarse_segm_to_mask,
)

ToMaskConverter.register(
    DensePoseChartPredictorOutput, predictor_output_with_fine_and_coarse_segm_to_mask
)

ToChartResultConverter.register(
    DensePoseChartPredictorOutput, densepose_chart_predictor_output_to_result
)

HFlipConverter.register(DensePoseChartPredictorOutput, densepose_chart_predictor_output_hflip)
