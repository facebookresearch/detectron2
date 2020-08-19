# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from detectron2.structures import BitMasks, BoxMode, Instances

from ..structures import resample_output_to_bbox


def densepose_to_mask(instances: Instances) -> BitMasks:
    """
    Produce masks from DensePose predictions
    DensePose predictions for a given image, stored in `pred_densepose` field,
    are instances of DensePoseOutput. This sampler takes
    `S` and `I` output tensors (coarse and fine segmentation) and converts
    then to a mask tensor, which is a bool tensor of the size of the input
    image

    Args:
        instances (Instances): predicted results, expected to have `pred_densepose` field
            that contains `DensePoseOutput` objects

    Returns:
        `BitMasks` instance with boolean tensors of the size of the input image that have non-zero
            values at pixels that are estimated to belong to the detected objects
    """
    H, W = instances.image_size
    boxes_xyxy_abs = instances.pred_boxes.tensor.clone().cpu()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    N = len(boxes_xywh_abs)
    gt_masks = torch.zeros((N, H, W), dtype=torch.bool, device=torch.device("cpu"))
    for i, box_xywh in enumerate(boxes_xywh_abs):
        labels_i, _ = resample_output_to_bbox(instances.pred_densepose[i], box_xywh)
        x, y, w, h = box_xywh.long().tolist()
        gt_masks[i, y : y + h, x : x + w] = labels_i.cpu() > 0
    return BitMasks(gt_masks)


class MaskFromDensePoseSampler:
    """
    Produce mask GT from DensePose predictions
    DensePose prediction is an instance of DensePoseOutput. This sampler takes
    `S` and `I` output tensors (coarse and fine segmentation) and converts
    then to a mask tensor, which is a bool tensor of the size of the input
    image
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
        return densepose_to_mask(instances)
