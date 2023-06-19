# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any

from .base import BaseConverter


class HFlipConverter(BaseConverter):
    """
    Converts various DensePose predictor outputs to DensePose results.
    Each DensePose predictor output type has to register its convertion strategy.
    """

    registry = {}
    dst_type = None

    @classmethod
    # pyre-fixme[14]: `convert` overrides method defined in `BaseConverter`
    #  inconsistently.
    def convert(cls, predictor_outputs: Any, transform_data: Any, *args, **kwargs):
        """
        Performs an horizontal flip on DensePose predictor outputs.
        Does recursive lookup for base classes, so there's no need
        for explicit registration for derived classes.

        Args:
            predictor_outputs: DensePose predictor output to be converted to BitMasks
            transform_data: Anything useful for the flip
        Return:
            An instance of the same type as predictor_outputs
        """
        return super(HFlipConverter, cls).convert(
            predictor_outputs, transform_data, *args, **kwargs
        )
