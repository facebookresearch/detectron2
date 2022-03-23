# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any
import torch
from torch.nn import functional as F

from detectron2.structures import BitMasks, Boxes, BoxMode

from .base import IntTupleBox, make_int_box
from .to_mask import ImageSizeType


def resample_coarse_segm_tensor_to_bbox(coarse_segm: torch.Tensor, box_xywh_abs: IntTupleBox):
    """
    Resample coarse segmentation tensor to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    labels = F.interpolate(coarse_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
    return labels


def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs: IntTupleBox
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm, (h, w), mode="bilinear", align_corners=False
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        * (coarse_segm_bbox > 0).long()
    )
    return labels


def resample_fine_and_coarse_segm_to_bbox(predictor_output: Any, box_xywh_abs: IntTupleBox):
    """
    Resample fine and coarse segmentation outputs from a predictor to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        predictor_output: DensePose predictor output that contains segmentation
            results to be resampled
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    return resample_fine_and_coarse_segm_tensors_to_bbox(
        predictor_output.fine_segm,
        predictor_output.coarse_segm,
        box_xywh_abs,
    )


def predictor_output_with_coarse_segm_to_mask(
    predictor_output: Any, boxes: Boxes, image_size_hw: ImageSizeType
) -> BitMasks:
    """
    Convert predictor output with coarse and fine segmentation to a mask.
    Assumes that predictor output has the following attributes:
     - coarse_segm (tensor of size [N, D, H, W]): coarse segmentation
         unnormalized scores for N instances; D is the number of coarse
         segmentation labels, H and W is the resolution of the estimate

    Args:
        predictor_output: DensePose predictor output to be converted to mask
        boxes (Boxes): bounding boxes that correspond to the DensePose
            predictor outputs
        image_size_hw (tuple [int, int]): image height Himg and width Wimg
    Return:
        BitMasks that contain a bool tensor of size [N, Himg, Wimg] with
        a mask of the size of the image for each instance
    """
    H, W = image_size_hw
    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    N = len(boxes_xywh_abs)
    masks = torch.zeros((N, H, W), dtype=torch.bool, device=boxes.tensor.device)
    for i in range(len(boxes_xywh_abs)):
        box_xywh = make_int_box(boxes_xywh_abs[i])
        box_mask = resample_coarse_segm_tensor_to_bbox(predictor_output[i].coarse_segm, box_xywh)
        x, y, w, h = box_xywh
        masks[i, y : y + h, x : x + w] = box_mask

    return BitMasks(masks)


def predictor_output_with_fine_and_coarse_segm_to_mask(
    predictor_output: Any, boxes: Boxes, image_size_hw: ImageSizeType
) -> BitMasks:
    """
    Convert predictor output with coarse and fine segmentation to a mask.
    Assumes that predictor output has the following attributes:
     - coarse_segm (tensor of size [N, D, H, W]): coarse segmentation
         unnormalized scores for N instances; D is the number of coarse
         segmentation labels, H and W is the resolution of the estimate
     - fine_segm (tensor of size [N, C, H, W]): fine segmentation
         unnormalized scores for N instances; C is the number of fine
         segmentation labels, H and W is the resolution of the estimate

    Args:
        predictor_output: DensePose predictor output to be converted to mask
        boxes (Boxes): bounding boxes that correspond to the DensePose
            predictor outputs
        image_size_hw (tuple [int, int]): image height Himg and width Wimg
    Return:
        BitMasks that contain a bool tensor of size [N, Himg, Wimg] with
        a mask of the size of the image for each instance
    """
    H, W = image_size_hw
    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    N = len(boxes_xywh_abs)
    masks = torch.zeros((N, H, W), dtype=torch.bool, device=boxes.tensor.device)
    for i in range(len(boxes_xywh_abs)):
        box_xywh = make_int_box(boxes_xywh_abs[i])
        labels_i = resample_fine_and_coarse_segm_to_bbox(predictor_output[i], box_xywh)
        x, y, w, h = box_xywh
        masks[i, y : y + h, x : x + w] = labels_i > 0
    return BitMasks(masks)
