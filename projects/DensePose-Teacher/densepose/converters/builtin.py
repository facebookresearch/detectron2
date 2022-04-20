# Copyright (c) Facebook, Inc. and its affiliates.

from ..structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from . import (
    HFlipConverter,
    ToChartResultConverter,
    ToChartResultConverterWithConfidences,
    ToMaskConverter,
    densepose_chart_predictor_output_hflip,
    densepose_chart_predictor_output_to_result,
    densepose_chart_predictor_output_to_result_with_confidences,
    predictor_output_with_coarse_segm_to_mask,
    predictor_output_with_fine_and_coarse_segm_to_mask,
)

ToMaskConverter.register(
    DensePoseChartPredictorOutput, predictor_output_with_fine_and_coarse_segm_to_mask
)
ToMaskConverter.register(
    DensePoseEmbeddingPredictorOutput, predictor_output_with_coarse_segm_to_mask
)

ToChartResultConverter.register(
    DensePoseChartPredictorOutput, densepose_chart_predictor_output_to_result
)

ToChartResultConverterWithConfidences.register(
    DensePoseChartPredictorOutput, densepose_chart_predictor_output_to_result_with_confidences
)

HFlipConverter.register(DensePoseChartPredictorOutput, densepose_chart_predictor_output_hflip)
