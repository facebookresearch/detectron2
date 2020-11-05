# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List
import torch
from torch.nn import functional as F

from detectron2.structures import Instances

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


class SingleTensorsHelper:
    """
    Helper class that holds data related to tensor extraction and packing,
    required for more efficient loss computation. The input data that contains
    matches between detections and ground truth is organized on per-image basis.
    Matches for a single image are stored in an `Instances` object. Not all the
    matches contain ground truth for DensePose. This we need to extract the
    relevant data and pack it in single tensors for more efficient loss
    computation.

    Assume one gets `K` `Instances` objects as an input with `N_1`, `N_2` ...
    `N_K` detection / ground truth matches respectively. Let's assume
    all of them contain the `gt_densepose` attribute (those that do not have it
    are skipped and not taken into consideration).

    Then this class defines the following attributes:
    * index_img (list of int): list of length M that stores image index `k` for all
        matches with valid DensePose annotations (takes values 0 .. K-1)
    * index_with_dp (list of int): list of length M that stores global indices of
        matches that have valid DensePose anotations (takes values 0 .. sum(N_k) - 1)
    * bbox_xywh_est (tensor of size [M, 4]): detected bounding boxes in XYWH format
        for matches that have valid DensePose anotations
    * bbox_xywh_gt (tensor of size [M, 4]): ground truth boxes in XYWH format for
        for matches that have valid DensePose anotations
    * fine_segm_labels_gt (tensor of size [J] of long): ground truth fine segmentation
        labels for annotated points; for each match `m` there is a certain number of
        annotated points `j_m`. Thus `J=sum(j_m)` where `m = 0 .. M - 1`
    * x_norm (tensor of size [J] of float): ground truth X normalized coordinates
        for annotated points;
    * y_norm (tensor of size [J] of float): ground truth Y normalized coordinates
        for annotated points;
    * u_gt (tensor of size [J] of float): ground truth U coordinates for annotated points;
    * v_gt (tensor of size [J] of float): ground truth V coordinates for annotated points;
    * coarse_segm_gt (tensor of size [M, S, S] of long): ground truth coarse segmentation,
        one tensor of size [S, S] for each ground truth bounding box;
    * index_bbox (tensor of size [J] of long): contains match index `m` for each annotated
        point, `m` takes values `0 .. M - 1`
    """

    def __init__(self, proposals_with_gt: List[Instances]):

        with torch.no_grad():
            (
                index_img,
                index_with_dp,
                bbox_xywh_est,
                bbox_xywh_gt,
                fine_segm_labels_gt,
                x_norm,
                y_norm,
                u_gt,
                v_gt,
                coarse_segm_gt,
                index_bbox,
            ) = _extract_single_tensors_from_matches(proposals_with_gt)

        for k, v in locals().items():
            if k not in ["self", "proposals_with_gt"]:
                setattr(self, k, v)


class BilinearInterpolationHelper:
    """
    Args:
        tensors_helper (SingleTensorsHelper)
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
        tensors_helper: SingleTensorsHelper,
        j_valid,
        y_lo,
        y_hi,
        x_lo,
        x_hi,
        w_ylo_xlo,
        w_ylo_xhi,
        w_yhi_xlo,
        w_yhi_xhi,
    ):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)

    @staticmethod
    def from_matches(tensors_helper: SingleTensorsHelper, densepose_outputs_size):

        zh, zw = densepose_outputs_size[2], densepose_outputs_size[3]

        x0_gt, y0_gt, w_gt, h_gt = tensors_helper.bbox_xywh_gt[tensors_helper.index_bbox].unbind(1)
        x0_est, y0_est, w_est, h_est = tensors_helper.bbox_xywh_est[
            tensors_helper.index_bbox
        ].unbind(dim=1)
        x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
            tensors_helper.x_norm, x0_gt, w_gt, x0_est, w_est, zw
        )
        y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
            tensors_helper.y_norm, y0_gt, h_gt, y0_est, h_est, zh
        )
        j_valid = jx_valid * jy_valid

        w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
        w_ylo_xhi = x_w * (1.0 - y_w)
        w_yhi_xlo = (1.0 - x_w) * y_w
        w_yhi_xhi = x_w * y_w

        return BilinearInterpolationHelper(
            tensors_helper,
            j_valid,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
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
            self.tensors_helper.fine_segm_labels_gt if slice_fine_segm is None else slice_fine_segm
        )
        w_ylo_xlo = self.w_ylo_xlo if w_ylo_xlo is None else w_ylo_xlo
        w_ylo_xhi = self.w_ylo_xhi if w_ylo_xhi is None else w_ylo_xhi
        w_yhi_xlo = self.w_yhi_xlo if w_yhi_xlo is None else w_yhi_xlo
        w_yhi_xhi = self.w_yhi_xhi if w_yhi_xhi is None else w_yhi_xhi

        index_bbox = self.tensors_helper.index_bbox
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


def _extract_single_tensors_from_matches_one_image(
    proposals_targets, bbox_with_dp_offset, bbox_global_offset
):
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    # Ibbox_all == k should be true for all data that corresponds
    # to bbox_xywh_gt[k] and bbox_xywh_est[k]
    # index k here is global wrt images
    i_bbox_all = []
    # at offset k (k is global) contains index of bounding box data
    # within densepose output tensor
    i_with_dp = []

    boxes_xywh_est = proposals_targets.proposal_boxes.clone()
    boxes_xywh_gt = proposals_targets.gt_boxes.clone()
    n_i = len(boxes_xywh_est)
    assert n_i == len(boxes_xywh_gt)

    if n_i:
        boxes_xywh_est.tensor[:, 2] -= boxes_xywh_est.tensor[:, 0]
        boxes_xywh_est.tensor[:, 3] -= boxes_xywh_est.tensor[:, 1]
        boxes_xywh_gt.tensor[:, 2] -= boxes_xywh_gt.tensor[:, 0]
        boxes_xywh_gt.tensor[:, 3] -= boxes_xywh_gt.tensor[:, 1]
        if hasattr(proposals_targets, "gt_densepose"):
            densepose_gt = proposals_targets.gt_densepose
            for k, box_xywh_est, box_xywh_gt, dp_gt in zip(
                range(n_i), boxes_xywh_est.tensor, boxes_xywh_gt.tensor, densepose_gt
            ):
                if (dp_gt is not None) and (len(dp_gt.x) > 0):
                    i_gt_all.append(dp_gt.i)
                    x_norm_all.append(dp_gt.x)
                    y_norm_all.append(dp_gt.y)
                    u_gt_all.append(dp_gt.u)
                    v_gt_all.append(dp_gt.v)
                    s_gt_all.append(dp_gt.segm.unsqueeze(0))
                    bbox_xywh_gt_all.append(box_xywh_gt.view(-1, 4))
                    bbox_xywh_est_all.append(box_xywh_est.view(-1, 4))
                    i_bbox_k = torch.full_like(dp_gt.i, bbox_with_dp_offset + len(i_with_dp))
                    i_bbox_all.append(i_bbox_k)
                    i_with_dp.append(bbox_global_offset + k)
    return (
        i_gt_all,
        x_norm_all,
        y_norm_all,
        u_gt_all,
        v_gt_all,
        s_gt_all,
        bbox_xywh_gt_all,
        bbox_xywh_est_all,
        i_bbox_all,
        i_with_dp,
    )


def _extract_single_tensors_from_matches(proposals_with_targets: List[Instances]):
    i_img = []
    i_gt_all = []
    x_norm_all = []
    y_norm_all = []
    u_gt_all = []
    v_gt_all = []
    s_gt_all = []
    bbox_xywh_gt_all = []
    bbox_xywh_est_all = []
    i_bbox_all = []
    i_with_dp_all = []
    n = 0
    for i, proposals_targets_per_image in enumerate(proposals_with_targets):
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        (
            i_gt_img,
            x_norm_img,
            y_norm_img,
            u_gt_img,
            v_gt_img,
            s_gt_img,
            bbox_xywh_gt_img,
            bbox_xywh_est_img,
            i_bbox_img,
            i_with_dp_img,
        ) = _extract_single_tensors_from_matches_one_image(  # noqa
            proposals_targets_per_image, len(i_with_dp_all), n
        )
        i_gt_all.extend(i_gt_img)
        x_norm_all.extend(x_norm_img)
        y_norm_all.extend(y_norm_img)
        u_gt_all.extend(u_gt_img)
        v_gt_all.extend(v_gt_img)
        s_gt_all.extend(s_gt_img)
        bbox_xywh_gt_all.extend(bbox_xywh_gt_img)
        bbox_xywh_est_all.extend(bbox_xywh_est_img)
        i_bbox_all.extend(i_bbox_img)
        i_with_dp_all.extend(i_with_dp_img)
        i_img.extend([i] * len(i_with_dp_img))
        n += n_i
    # concatenate all data into a single tensor
    if (n > 0) and (len(i_with_dp_all) > 0):
        i_gt = torch.cat(i_gt_all, 0).long()
        x_norm = torch.cat(x_norm_all, 0)
        y_norm = torch.cat(y_norm_all, 0)
        u_gt = torch.cat(u_gt_all, 0)
        v_gt = torch.cat(v_gt_all, 0)
        s_gt = torch.cat(s_gt_all, 0)
        bbox_xywh_gt = torch.cat(bbox_xywh_gt_all, 0)
        bbox_xywh_est = torch.cat(bbox_xywh_est_all, 0)
        i_bbox = torch.cat(i_bbox_all, 0).long()
    else:
        i_gt = None
        x_norm = None
        y_norm = None
        u_gt = None
        v_gt = None
        s_gt = None
        bbox_xywh_gt = None
        bbox_xywh_est = None
        i_bbox = None
    return (
        i_img,
        i_with_dp_all,
        bbox_xywh_est,
        bbox_xywh_gt,
        i_gt,
        x_norm,
        y_norm,
        u_gt,
        v_gt,
        s_gt,
        i_bbox,
    )
