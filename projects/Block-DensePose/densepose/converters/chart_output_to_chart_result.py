# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict
import torch
import numpy as np
from torch.nn import functional as F

from detectron2.structures.boxes import Boxes, BoxMode

from ..structures import (
    DensePoseChartPredictorOutput,
    DensePoseChartResult,
    DensePoseChartResultWithConfidences,
    BlockPredictorOutput,
    DistributionPredictorOutput,
)
from . import resample_fine_and_coarse_segm_to_bbox
from .base import IntTupleBox, make_int_box


def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # pyre-fixme[6]: Expected `Optional[int]` for 2nd param but got `Tuple[int, int]`.
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    # pyre-fixme[6]: Expected `Optional[int]` for 2nd param but got `Tuple[int, int]`.
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv


def resample_uv_to_bbox(
    predictor_output: DensePoseChartPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be resampled
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    return resample_uv_tensors_to_bbox(
        predictor_output.u,
        predictor_output.v,
        labels,
        box_xywh_abs,
    )


def resample_block_uv_to_bbox(
    predictor_output: BlockPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
    block_num: int,
    analysis=False
):
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be resampled
        labels (tensor [H, W] of uint8): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    # block_fpath = cfg.MODEL.ROI_DENSEPOSE_HEAD.block_FPATH
    block_width = 1. / block_num
    block_center = torch.linspace(block_width / 2, 1. - block_width / 2, block_num)

    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)

    u_cls = F.interpolate(predictor_output.u_cls, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num, h, w).permute(0, 2, 3, 1)
    u_offset = F.interpolate(predictor_output.u_offset, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num, h, w).permute(0, 2, 3, 1)
    v_cls = F.interpolate(predictor_output.v_cls, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num, h, w).permute(0, 2, 3, 1)
    v_offset = F.interpolate(predictor_output.v_offset, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num, h, w).permute(0, 2, 3, 1)

    uv = torch.zeros([2, h, w], dtype=torch.float32, device=predictor_output.u_cls.device)
    conf_u, index_u = torch.max(F.softmax(u_cls, dim=3), dim=3)
    conf_v, index_v = torch.max(F.softmax(v_cls, dim=3), dim=3)
    conf = torch.zeros([2, h, w], dtype=torch.float32, device=predictor_output.u_cls.device)

    for part_id in range(1, 25):
        uv[0][labels == part_id] = u_offset[part_id-1][labels == part_id, index_u[part_id-1, (labels == part_id)]] * \
                                   block_width + block_center[index_u[part_id-1][(labels == part_id)]]
        uv[1][labels == part_id] = v_offset[part_id-1][labels == part_id, index_v[part_id-1, (labels == part_id)]] * \
                                   block_width + block_center[index_v[part_id-1][(labels == part_id)]]
        conf[0][labels == part_id] = conf_u[part_id-1][labels == part_id]
        conf[1][labels == part_id] = conf_v[part_id-1][labels == part_id]

    if analysis:
        return uv, torch.sqrt(conf[0] * conf[1]).unsqueeze(0)
    return uv, None


def resample_distribution_uv_to_bbox(
    predictor_output: DistributionPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
    block_num: int
) -> torch.Tensor:
    block_value = torch.linspace(0, block_num, block_num + 1)
    block_value *= 1. / block_num

    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)

    u_dis = F.interpolate(predictor_output.u_dis, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num + 1, h, w).permute(0, 2, 3, 1)
    v_dis = F.interpolate(predictor_output.v_dis, (h, w), mode="bilinear",
                          align_corners=False)[0].reshape(24, block_num + 1, h, w).permute(0, 2, 3, 1)

    uv = torch.zeros([2, h, w], dtype=torch.float32, device=predictor_output.u_dis.device)

    for part_id in range(1, 25):
        uv[0][labels == part_id] = F.linear(F.softmax(u_dis[part_id - 1][labels == part_id], dim=1),
                                            block_value.float())
        uv[1][labels == part_id] = F.linear(F.softmax(v_dis[part_id - 1][labels == part_id], dim=1),
                                            block_value.float())

    return uv


def densepose_chart_predictor_output_to_result(
    predictor_output: DensePoseChartPredictorOutput, boxes: Boxes
) -> DensePoseChartResult:
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result (DensePoseChartResult)
    """
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv = resample_uv_to_bbox(predictor_output, labels, box_xywh)
    return DensePoseChartResult(labels=labels, uv=uv)


def block_predictor_output_to_result(
    predictor_output: BlockPredictorOutput, boxes: Boxes,
    *args, **kwargs
):
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result (DensePoseChartResult)
    """
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv, conf_uv = resample_block_uv_to_bbox(predictor_output, labels, box_xywh, kwargs["block_num"], analysis=kwargs["analysis"])
    return DensePoseChartResult(labels=labels, uv=uv), conf_uv


def distribution_predictor_output_to_result(
    predictor_output: DistributionPredictorOutput, boxes: Boxes,
    *args, **kwargs
) -> DensePoseChartResult:
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv = resample_distribution_uv_to_bbox(predictor_output, labels, box_xywh, kwargs["block_num"])
    return DensePoseChartResult(labels=labels, uv=uv)


def resample_confidences_to_bbox(
    predictor_output: DensePoseChartPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> Dict[str, torch.Tensor]:
    """
    Resamples confidences for the given bounding box

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be resampled
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled confidences - a dict of [H, W] tensors of float
    """

    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)

    confidence_names = [
        "sigma_1",
        "sigma_2",
        "kappa_u",
        "kappa_v",
        "fine_segm_confidence",
        "coarse_segm_confidence",
    ]
    confidence_results = {key: None for key in confidence_names}
    confidence_names = [
        key for key in confidence_names if getattr(predictor_output, key) is not None
    ]
    confidence_base = torch.zeros([h, w], dtype=torch.float32, device=predictor_output.u.device)

    # assign data from channels that correspond to the labels
    for key in confidence_names:
        resampled_confidence = F.interpolate(
            # pyre-fixme[6]: Expected `Optional[int]` for 2nd param but got
            #  `Tuple[int, int]`.
            getattr(predictor_output, key), (h, w), mode="bilinear", align_corners=False
        )
        result = confidence_base.clone()
        for part_id in range(1, predictor_output.u.size(1)):
            if resampled_confidence.size(1) != predictor_output.u.size(1):
                # confidence is not part-based, don't try to fill it part by part
                continue
            result[labels == part_id] = resampled_confidence[0, part_id][labels == part_id]

        if resampled_confidence.size(1) != predictor_output.u.size(1):
            # confidence is not part-based, fill the data with the first channel
            # (targeted for segmentation confidences that have only 1 channel)
            result = resampled_confidence[0, 0]

        confidence_results[key] = result

    return confidence_results  # pyre-ignore[7]


def densepose_chart_predictor_output_to_result_with_confidences(
    predictor_output: DensePoseChartPredictorOutput, boxes: Boxes
) -> DensePoseChartResultWithConfidences:
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output with confidences to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result with confidences (DensePoseChartResultWithConfidences)
    """
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv = resample_uv_to_bbox(predictor_output, labels, box_xywh)
    confidences = resample_confidences_to_bbox(predictor_output, labels, box_xywh)
    return DensePoseChartResultWithConfidences(labels=labels, uv=uv, **confidences)
