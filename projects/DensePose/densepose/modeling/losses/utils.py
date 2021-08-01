# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn import functional as F

from detectron2.structures import BoxMode, Instances

from densepose import DensePoseDataRelative

LossDict = Dict[str, torch.Tensor]


def _linear_interpolation_utilities(v_norm, v0_src, size_src, v0_dst, size_dst, size_z):
    """
    Computes utility values for linear interpolation at points v.
    The points are given as normalized offsets in the source interval
    (v0_src, v0_src + size_src), more precisely:
        v = v0_src + v_norm * size_src / 256.0
    The computed utilities include lower points v_lo, upper points v_hi,
    interpolation weights v_w and flags j_valid indicating whether the
    points falls into the destination interval (v0_dst, v0_dst + size_dst).

    Args:
        v_norm (:obj: `torch.Tensor`): tensor of size N containing
            normalized point offsets
        v0_src (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of source intervals for normalized points
        size_src (:obj: `torch.Tensor`): tensor of size N containing
            source interval sizes for normalized points
        v0_dst (:obj: `torch.Tensor`): tensor of size N containing
            left bounds of destination intervals
        size_dst (:obj: `torch.Tensor`): tensor of size N containing
            destination interval sizes
        size_z (int): interval size for data to be interpolated

    Returns:
        v_lo (:obj: `torch.Tensor`): int tensor of size N containing
            indices of lower values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_hi (:obj: `torch.Tensor`): int tensor of size N containing
            indices of upper values used for interpolation, all values are
            integers from [0, size_z - 1]
        v_w (:obj: `torch.Tensor`): float tensor of size N containing
            interpolation weights
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size N containing
            0 for points outside the estimation interval
            (v0_est, v0_est + size_est) and 1 otherwise
    """
    v = v0_src + v_norm * size_src / 256.0
    j_valid = (v - v0_dst >= 0) * (v - v0_dst < size_dst)
    v_grid = (v - v0_dst) * size_z / size_dst
    v_lo = v_grid.floor().long().clamp(min=0, max=size_z - 1)
    v_hi = (v_lo + 1).clamp(max=size_z - 1)
    v_grid = torch.min(v_hi.float(), v_grid)
    v_w = v_grid - v_lo.float()
    return v_lo, v_hi, v_w, j_valid


class BilinearInterpolationHelper:
    """
    Args:
        packed_annotations: object that contains packed annotations
        j_valid (:obj: `torch.Tensor`): uint8 tensor of size M containing
            0 for points to be discarded and 1 for points to be selected
        y_lo (:obj: `torch.Tensor`): int tensor of indices of upper values
            in z_est for each point
        y_hi (:obj: `torch.Tensor`): int tensor of indices of lower values
            in z_est for each point
        x_lo (:obj: `torch.Tensor`): int tensor of indices of left values
            in z_est for each point
        x_hi (:obj: `torch.Tensor`): int tensor of indices of right values
            in z_est for each point
        w_ylo_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-left value weight for each point
        w_ylo_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains upper-right value weight for each point
        w_yhi_xlo (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-left value weight for each point
        w_yhi_xhi (:obj: `torch.Tensor`): float tensor of size M;
            contains lower-right value weight for each point
    """

    def __init__(
        self,
        packed_annotations: Any,
        j_valid: torch.Tensor,
        y_lo: torch.Tensor,
        y_hi: torch.Tensor,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        w_ylo_xlo: torch.Tensor,
        w_ylo_xhi: torch.Tensor,
        w_yhi_xlo: torch.Tensor,
        w_yhi_xhi: torch.Tensor,
    ):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)

    @staticmethod
    def from_matches(
        packed_annotations: Any, densepose_outputs_size_hw: Tuple[int, int]
    ) -> "BilinearInterpolationHelper":
        """
        Args:
            packed_annotations: annotations packed into tensors, the following
                attributes are required:
                 - bbox_xywh_gt
                 - bbox_xywh_est
                 - x_gt
                 - y_gt
                 - point_bbox_with_dp_indices
                 - point_bbox_indices
            densepose_outputs_size_hw (tuple [int, int]): resolution of
                DensePose predictor outputs (H, W)
        Return:
            An instance of `BilinearInterpolationHelper` used to perform
            interpolation for the given annotation points and output resolution
        """

        zh, zw = densepose_outputs_size_hw
        x0_gt, y0_gt, w_gt, h_gt = packed_annotations.bbox_xywh_gt[
            packed_annotations.point_bbox_with_dp_indices
        ].unbind(dim=1)
        x0_est, y0_est, w_est, h_est = packed_annotations.bbox_xywh_est[
            packed_annotations.point_bbox_with_dp_indices
        ].unbind(dim=1)
        x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
            packed_annotations.x_gt, x0_gt, w_gt, x0_est, w_est, zw
        )
        y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
            packed_annotations.y_gt, y0_gt, h_gt, y0_est, h_est, zh
        )
        j_valid = jx_valid * jy_valid

        w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
        w_ylo_xhi = x_w * (1.0 - y_w)
        w_yhi_xlo = (1.0 - x_w) * y_w
        w_yhi_xhi = x_w * y_w

        return BilinearInterpolationHelper(
            packed_annotations,
            j_valid,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,  # pyre-ignore[6]
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )

    def extract_at_points(
        self,
        z_est,
        slice_fine_segm=None,
        w_ylo_xlo=None,
        w_ylo_xhi=None,
        w_yhi_xlo=None,
        w_yhi_xhi=None,
    ):
        """
        Extract ground truth values z_gt for valid point indices and estimated
        values z_est using bilinear interpolation over top-left (y_lo, x_lo),
        top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
        (y_hi, x_hi) values in z_est with corresponding weights:
        w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
        Use slice_fine_segm to slice dim=1 in z_est
        """
        slice_fine_segm = (
            self.packed_annotations.fine_segm_labels_gt
            if slice_fine_segm is None
            else slice_fine_segm
        )
        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        index_bbox = self.packed_annotations.point_bbox_indices
        z_est_sampled = (
            z_est[index_bbox, slice_fine_segm, self.y_lo, self.x_lo] * w_ylo_xlo
            + z_est[index_bbox, slice_fine_segm, self.y_lo, self.x_hi] * w_ylo_xhi
            + z_est[index_bbox, slice_fine_segm, self.y_hi, self.x_lo] * w_yhi_xlo
            + z_est[index_bbox, slice_fine_segm, self.y_hi, self.x_hi] * w_yhi_xhi
        )
        return z_est_sampled


def resample_data(
    z, bbox_xywh_src, bbox_xywh_dst, wout, hout, mode="nearest", padding_mode="zeros"
):
    """
    Args:
        z (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with data to be
            resampled
        bbox_xywh_src (:obj: `torch.Tensor`): tensor of size (N,4) containing
            source bounding boxes in format XYWH
        bbox_xywh_dst (:obj: `torch.Tensor`): tensor of size (N,4) containing
            destination bounding boxes in format XYWH
    Return:
        zresampled (:obj: `torch.Tensor`): tensor of size (N, C, Hout, Wout)
            with resampled values of z, where D is the discretization size
    """
    n = bbox_xywh_src.size(0)
    assert n == bbox_xywh_dst.size(0), (
        "The number of "
        "source ROIs for resampling ({}) should be equal to the number "
        "of destination ROIs ({})".format(bbox_xywh_src.size(0), bbox_xywh_dst.size(0))
    )
    x0src, y0src, wsrc, hsrc = bbox_xywh_src.unbind(dim=1)
    x0dst, y0dst, wdst, hdst = bbox_xywh_dst.unbind(dim=1)
    x0dst_norm = 2 * (x0dst - x0src) / wsrc - 1
    y0dst_norm = 2 * (y0dst - y0src) / hsrc - 1
    x1dst_norm = 2 * (x0dst + wdst - x0src) / wsrc - 1
    y1dst_norm = 2 * (y0dst + hdst - y0src) / hsrc - 1
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w_expanded = grid_w[None, None, :].expand(n, hout, wout)
    grid_h_expanded = grid_h[None, :, None].expand(n, hout, wout)
    dx_expanded = (x1dst_norm - x0dst_norm)[:, None, None].expand(n, hout, wout)
    dy_expanded = (y1dst_norm - y0dst_norm)[:, None, None].expand(n, hout, wout)
    x0_expanded = x0dst_norm[:, None, None].expand(n, hout, wout)
    y0_expanded = y0dst_norm[:, None, None].expand(n, hout, wout)
    grid_x = grid_w_expanded * dx_expanded + x0_expanded
    grid_y = grid_h_expanded * dy_expanded + y0_expanded
    grid = torch.stack((grid_x, grid_y), dim=3)
    # resample Z from (N, C, H, W) into (N, C, Hout, Wout)
    zresampled = F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return zresampled


class AnnotationsAccumulator(ABC):
    """
    Abstract class for an accumulator for annotations that can produce
    dense annotations packed into tensors.
    """

    @abstractmethod
    def accumulate(self, instances_one_image: Instances):
        """
        Accumulate instances data for one image

        Args:
            instances_one_image (Instances): instances data to accumulate
        """
        pass

    @abstractmethod
    def pack(self) -> Any:
        """
        Pack data into tensors
        """
        pass


@dataclass
class PackedChartBasedAnnotations:
    """
    Packed annotations for chart-based model training. The following attributes
    are defined:
     - fine_segm_labels_gt (tensor [K] of `int64`): GT fine segmentation point labels
     - x_gt (tensor [K] of `float32`): GT normalized X point coordinates
     - y_gt (tensor [K] of `float32`): GT normalized Y point coordinates
     - u_gt (tensor [K] of `float32`): GT point U values
     - v_gt (tensor [K] of `float32`): GT point V values
     - coarse_segm_gt (tensor [N, S, S] of `float32`): GT segmentation for bounding boxes
     - bbox_xywh_gt (tensor [N, 4] of `float32`): selected GT bounding boxes in
         XYWH format
     - bbox_xywh_est (tensor [N, 4] of `float32`): selected matching estimated
         bounding boxes in XYWH format
     - point_bbox_with_dp_indices (tensor [K] of `int64`): indices of bounding boxes
         with DensePose annotations that correspond to the point data
     - point_bbox_indices (tensor [K] of `int64`): indices of bounding boxes
         (not necessarily the selected ones with DensePose data) that correspond
         to the point data
     - bbox_indices (tensor [N] of `int64`): global indices of selected bounding
         boxes with DensePose annotations; these indices could be used to access
         features that are computed for all bounding boxes, not only the ones with
         DensePose annotations.
    Here K is the total number of points and N is the total number of instances
    with DensePose annotations.
    """

    fine_segm_labels_gt: torch.Tensor
    x_gt: torch.Tensor
    y_gt: torch.Tensor
    u_gt: torch.Tensor
    v_gt: torch.Tensor
    coarse_segm_gt: Optional[torch.Tensor]
    bbox_xywh_gt: torch.Tensor
    bbox_xywh_est: torch.Tensor
    point_bbox_with_dp_indices: torch.Tensor
    point_bbox_indices: torch.Tensor
    bbox_indices: torch.Tensor


class ChartBasedAnnotationsAccumulator(AnnotationsAccumulator):
    """
    Accumulates annotations by batches that correspond to objects detected on
    individual images. Can pack them together into single tensors.
    """

    def __init__(self):
        self.i_gt = []
        self.x_gt = []
        self.y_gt = []
        self.u_gt = []
        self.v_gt = []
        self.s_gt = []
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
                self._do_accumulate(box_xywh_gt, box_xywh_est, dp_gt)
            self.nxt_bbox_index += 1

    def _do_accumulate(
        self, box_xywh_gt: torch.Tensor, box_xywh_est: torch.Tensor, dp_gt: DensePoseDataRelative
    ):
        """
        Accumulate instances data for one image, given that the data is not empty

        Args:
            box_xywh_gt (tensor): GT bounding box
            box_xywh_est (tensor): estimated bounding box
            dp_gt (DensePoseDataRelative): GT densepose data
        """
        self.i_gt.append(dp_gt.i)
        self.x_gt.append(dp_gt.x)
        self.y_gt.append(dp_gt.y)
        self.u_gt.append(dp_gt.u)
        self.v_gt.append(dp_gt.v)
        if hasattr(dp_gt, "segm"):
            self.s_gt.append(dp_gt.segm.unsqueeze(0))
        self.bbox_xywh_gt.append(box_xywh_gt.view(-1, 4))
        self.bbox_xywh_est.append(box_xywh_est.view(-1, 4))
        self.point_bbox_with_dp_indices.append(
            torch.full_like(dp_gt.i, self.nxt_bbox_with_dp_index)
        )
        self.point_bbox_indices.append(torch.full_like(dp_gt.i, self.nxt_bbox_index))
        self.bbox_indices.append(self.nxt_bbox_index)
        self.nxt_bbox_with_dp_index += 1

    def pack(self) -> Optional[PackedChartBasedAnnotations]:
        """
        Pack data into tensors
        """
        if not len(self.i_gt):
            # TODO:
            # returning proper empty annotations would require
            # creating empty tensors of appropriate shape and
            # type on an appropriate device;
            # we return None so far to indicate empty annotations
            return None
        return PackedChartBasedAnnotations(
            fine_segm_labels_gt=torch.cat(self.i_gt, 0).long(),
            x_gt=torch.cat(self.x_gt, 0),
            y_gt=torch.cat(self.y_gt, 0),
            u_gt=torch.cat(self.u_gt, 0),
            v_gt=torch.cat(self.v_gt, 0),
            # ignore segmentation annotations, if not all the instances contain those
            coarse_segm_gt=torch.cat(self.s_gt, 0)
            if len(self.s_gt) == len(self.bbox_xywh_gt)
            else None,
            bbox_xywh_gt=torch.cat(self.bbox_xywh_gt, 0),
            bbox_xywh_est=torch.cat(self.bbox_xywh_est, 0),
            point_bbox_with_dp_indices=torch.cat(self.point_bbox_with_dp_indices, 0).long(),
            point_bbox_indices=torch.cat(self.point_bbox_indices, 0).long(),
            bbox_indices=torch.as_tensor(
                self.bbox_indices, dtype=torch.long, device=self.x_gt[0].device
            ).long(),
        )


def extract_packed_annotations_from_matches(
    proposals_with_targets: List[Instances], accumulator: AnnotationsAccumulator
) -> Any:
    for proposals_targets_per_image in proposals_with_targets:
        accumulator.accumulate(proposals_targets_per_image)
    return accumulator.pack()


def sample_random_indices(
    n_indices: int, n_samples: int, device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    """
    Samples `n_samples` random indices from range `[0..n_indices - 1]`.
    If `n_indices` is smaller than `n_samples`, returns `None` meaning that all indices
    are selected.
    Args:
        n_indices (int): total number of indices
        n_samples (int): number of indices to sample
        device (torch.device): the desired device of returned tensor
    Return:
        Tensor of selected vertex indices, or `None`, if all vertices are selected
    """
    if (n_samples <= 0) or (n_indices <= n_samples):
        return None
    indices = torch.randperm(n_indices, device=device)[:n_samples]
    return indices
