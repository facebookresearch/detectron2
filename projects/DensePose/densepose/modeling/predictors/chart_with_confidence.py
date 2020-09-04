# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import DensePoseChartConfidencePredictorMixin, DensePoseChartPredictor


class DensePoseChartWithConfidencePredictor(
    DensePoseChartConfidencePredictorMixin, DensePoseChartPredictor
):
    """
    Predictor that combines chart and chart confidence estimation
    """

    pass
