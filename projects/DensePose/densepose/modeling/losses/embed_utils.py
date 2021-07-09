# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import Any, Optional
import torch

from detectron2.structures import BoxMode, Instances

from .utils import AnnotationsAccumulator


@dataclass
class PackedCseAnnotations:
    x_gt: torch.Tensor
    y_gt: torch.Tensor
    coarse_segm_gt: Optional[torch.Tensor]
    vertex_mesh_ids_gt: torch.Tensor
    vertex_ids_gt: torch.Tensor
    bbox_xywh_gt: torch.Tensor
    bbox_xywh_est: torch.Tensor
    point_bbox_with_dp_indices: torch.Tensor
    point_bbox_indices: torch.Tensor
    bbox_indices: torch.Tensor


class CseAnnotationsAccumulator(AnnotationsAccumulator):
    """
    Accumulates annotations by batches that correspond to objects detected on
    individual images. Can pack them together into single tensors.
    """

    def __init__(self):
        self.x_gt = []
        self.y_gt = []
        self.s_gt = []
        self.vertex_mesh_ids_gt = []
        self.vertex_ids_gt = []
        self.bbox_xywh_gt = []
        self.bbox_xywh_est = []
        self.point_bbox_with_dp_indices = []
        self.point_bbox_indices = []
        self.bbox_indices = []
        self.nxt_bbox_with_dp_index = 0
        self.nxt_bbox_index = 0

    def accumulate(self, instances_one_image: Instances):
        """
        Accumulate instances data for one image

        Args:
            instances_one_image (Instances): instances data to accumulate
        """
        boxes_xywh_est = BoxMode.convert(
            instances_one_image.proposal_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )
        boxes_xywh_gt = BoxMode.convert(
            instances_one_image.gt_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )
        n_matches = len(boxes_xywh_gt)
        assert n_matches == len(
            boxes_xywh_est
        ), f"Got {len(boxes_xywh_est)} proposal boxes and {len(boxes_xywh_gt)} GT boxes"
        if not n_matches:
            # no detection - GT matches
            return
        if (
            not hasattr(instances_one_image, "gt_densepose")
            or instances_one_image.gt_densepose is None
        ):
            # no densepose GT for the detections, just increase the bbox index
            self.nxt_bbox_index += n_matches
            return
        for box_xywh_est, box_xywh_gt, dp_gt in zip(
            boxes_xywh_est, boxes_xywh_gt, instances_one_image.gt_densepose
        ):
            if (dp_gt is not None) and (len(dp_gt.x) > 0):
                self._do_accumulate(box_xywh_gt, box_xywh_est, dp_gt)  # pyre-ignore[6]
            self.nxt_bbox_index += 1

    def _do_accumulate(self, box_xywh_gt: torch.Tensor, box_xywh_est: torch.Tensor, dp_gt: Any):
        """
        Accumulate instances data for one image, given that the data is not empty

        Args:
            box_xywh_gt (tensor): GT bounding box
            box_xywh_est (tensor): estimated bounding box
            dp_gt: GT densepose data with the following attributes:
             - x: normalized X coordinates
             - y: normalized Y coordinates
             - segm: tensor of size [S, S] with coarse segmentation
             -
        """
        self.x_gt.append(dp_gt.x)
        self.y_gt.append(dp_gt.y)
        if hasattr(dp_gt, "segm"):
            self.s_gt.append(dp_gt.segm.unsqueeze(0))
        self.vertex_ids_gt.append(dp_gt.vertex_ids)
        self.vertex_mesh_ids_gt.append(torch.full_like(dp_gt.vertex_ids, dp_gt.mesh_id))
        self.bbox_xywh_gt.append(box_xywh_gt.view(-1, 4))
        self.bbox_xywh_est.append(box_xywh_est.view(-1, 4))
        self.point_bbox_with_dp_indices.append(
            torch.full_like(dp_gt.vertex_ids, self.nxt_bbox_with_dp_index)
        )
        self.point_bbox_indices.append(torch.full_like(dp_gt.vertex_ids, self.nxt_bbox_index))
        self.bbox_indices.append(self.nxt_bbox_index)
        self.nxt_bbox_with_dp_index += 1

    def pack(self) -> Optional[PackedCseAnnotations]:
        """
        Pack data into tensors
        """
        if not len(self.x_gt):
            # TODO:
            # returning proper empty annotations would require
            # creating empty tensors of appropriate shape and
            # type on an appropriate device;
            # we return None so far to indicate empty annotations
            return None
        return PackedCseAnnotations(
            x_gt=torch.cat(self.x_gt, 0),
            y_gt=torch.cat(self.y_gt, 0),
            vertex_mesh_ids_gt=torch.cat(self.vertex_mesh_ids_gt, 0),
            vertex_ids_gt=torch.cat(self.vertex_ids_gt, 0),
            # ignore segmentation annotations, if not all the instances contain those
            coarse_segm_gt=torch.cat(self.s_gt, 0)
            if len(self.s_gt) == len(self.bbox_xywh_gt)
            else None,
            bbox_xywh_gt=torch.cat(self.bbox_xywh_gt, 0),
            bbox_xywh_est=torch.cat(self.bbox_xywh_est, 0),
            point_bbox_with_dp_indices=torch.cat(self.point_bbox_with_dp_indices, 0),
            point_bbox_indices=torch.cat(self.point_bbox_indices, 0),
            bbox_indices=torch.as_tensor(
                self.bbox_indices, dtype=torch.long, device=self.x_gt[0].device
            ),
        )
