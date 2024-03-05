# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from typing import Any, Tuple

from detectron2.structures import BitMasks, Boxes

from .base import BaseConverter

ImageSizeType = Tuple[int, int]


class ToMaskConverter(BaseConverter):
    """
    Converts various DensePose predictor outputs to masks
    in bit mask format (see `BitMasks`). Each DensePose predictor output type
    has to register its convertion strategy.
    """

    registry = {}
    dst_type = BitMasks

    @classmethod
    # pyre-fixme[14]: `convert` overrides method defined in `BaseConverter`
    #  inconsistently.
    def convert(
        cls,
        densepose_predictor_outputs: Any,
        boxes: Boxes,
        image_size_hw: ImageSizeType,
        *args,
        **kwargs
    ) -> BitMasks:
        """
        Convert DensePose predictor outputs to BitMasks using some registered
        converter. Does recursive lookup for base classes, so there's no need
        for explicit registration for derived classes.

        Args:
            densepose_predictor_outputs: DensePose predictor output to be
                converted to BitMasks
            boxes (Boxes): bounding boxes that correspond to the DensePose
                predictor outputs
            image_size_hw (tuple [int, int]): image height and width
        Return:
            An instance of `BitMasks`. If no suitable converter was found, raises KeyError
        """
        return super(ToMaskConverter, cls).convert(
            densepose_predictor_outputs, boxes, image_size_hw, *args, **kwargs
        )
