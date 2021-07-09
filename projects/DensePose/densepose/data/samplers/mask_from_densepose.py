# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.structures import BitMasks, Instances

from densepose.converters import ToMaskConverter


class MaskFromDensePoseSampler:
    """
    Produce mask GT from DensePose predictions
    This sampler simply converts DensePose predictions to BitMasks
    that a contain a bool tensor of the size of the input image
    """

    def __call__(self, instances: Instances) -> BitMasks:
        """
        Converts predicted data from `instances` into the GT mask data

        Args:
            instances (Instances): predicted results, expected to have `pred_densepose` field

        Returns:
            Boolean Tensor of the size of the input image that has non-zero
            values at pixels that are estimated to belong to the detected object
        """
        return ToMaskConverter.convert(
            instances.pred_densepose, instances.pred_boxes, instances.image_size
        )
