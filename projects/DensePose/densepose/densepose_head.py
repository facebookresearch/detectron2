# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, interpolate
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.registry import Registry

from .structures import DensePoseOutput

ROI_DENSEPOSE_HEAD_REGISTRY = Registry("ROI_DENSEPOSE_HEAD")


def initialize_module_params(module):
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseDeepLabHead(nn.Module):
    def __init__(self, cfg, input_channels):
        super(DensePoseDeepLabHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        norm                 = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        self.use_nonlocal    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels

        self.ASPP = ASPP(input_channels, [6, 12, 56], n_channels)  # 6, 12, 56
        self.add_module("ASPP", self.ASPP)

        if self.use_nonlocal:
            self.NLBlock = NONLocalBlock2D(input_channels, bn_layer=True)
            self.add_module("NLBlock", self.NLBlock)
        # weight_init.c2_msra_fill(self.ASPP)

        for i in range(self.n_stacked_convs):
            norm_module = nn.GroupNorm(32, hidden_dim) if norm == "GN" else None
            layer = Conv2d(
                n_channels,
                hidden_dim,
                kernel_size,
                stride=1,
                padding=pad_size,
                bias=not norm,
                norm=norm_module,
            )
            weight_init.c2_msra_fill(layer)
            n_channels = hidden_dim
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
        self.n_out_channels = hidden_dim
        # initialize_module_params(self)

    def forward(self, features):
        x0 = features
        x = self.ASPP(x0)
        if self.use_nonlocal:
            x = self.NLBlock(x)
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


# Copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# See https://arxiv.org/pdf/1706.05587.pdf for details
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# copied from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_embedded_gaussian.py
# See https://arxiv.org/abs/1711.07971 for details
class _NonLocalBlockND(nn.Module):
    def __init__(
        self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim) #nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.GroupNorm  # (32, hidden_dim)nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(32, self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXHead(nn.Module):
    def __init__(self, cfg, input_channels):
        super(DensePoseV1ConvXHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

    def forward(self, features):
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name


class DensePosePredictor(nn.Module):
    def __init__(self, cfg, input_channels):

        super(DensePosePredictor, self).__init__()
        dim_in = input_channels
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.ann_index_lowres = ConvTranspose2d(
            dim_in, n_segm_chan, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.index_uv_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.u_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.v_lowres = ConvTranspose2d(
            dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def forward(self, head_outputs):
        ann_index_lowres = self.ann_index_lowres(head_outputs)
        index_uv_lowres = self.index_uv_lowres(head_outputs)
        u_lowres = self.u_lowres(head_outputs)
        v_lowres = self.v_lowres(head_outputs)

        def interp2d(input):
            return interpolate(
                input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )

        ann_index = interp2d(ann_index_lowres)
        index_uv = interp2d(index_uv_lowres)
        u = interp2d(u_lowres)
        v = interp2d(v_lowres)
        return (
            (ann_index, index_uv, u, v),
            (ann_index_lowres, index_uv_lowres, u_lowres, v_lowres),
        )


class DensePoseDataFilter(object):
    def __init__(self, cfg):
        self.iou_threshold = cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD

    @torch.no_grad()
    def __call__(self, proposals_with_targets):
        """
        Filters proposals with targets to keep only the ones relevant for
        DensePose training
        proposals: list(Instances), each element of the list corresponds to
            various instances (proposals, GT for boxes and densepose) for one
            image
        """
        proposals_filtered = []
        for proposals_per_image in proposals_with_targets:
            if not hasattr(proposals_per_image, "gt_densepose"):
                continue
            assert hasattr(proposals_per_image, "gt_boxes")
            assert hasattr(proposals_per_image, "proposal_boxes")
            gt_boxes = proposals_per_image.gt_boxes
            est_boxes = proposals_per_image.proposal_boxes
            # apply match threshold for densepose head
            iou = matched_boxlist_iou(gt_boxes, est_boxes)
            iou_select = iou > self.iou_threshold
            proposals_per_image = proposals_per_image[iou_select]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            # filter out any target without densepose annotation
            gt_densepose = proposals_per_image.gt_densepose
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            selected_indices = [
                i for i, dp_target in enumerate(gt_densepose) if dp_target is not None
            ]
            if len(selected_indices) != len(gt_densepose):
                proposals_per_image = proposals_per_image[selected_indices]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_densepose)
            proposals_filtered.append(proposals_per_image)
        return proposals_filtered


def build_densepose_head(cfg, input_channels):
    head_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME
    return ROI_DENSEPOSE_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


def build_densepose_predictor(cfg, input_channels):
    predictor = DensePosePredictor(cfg, input_channels)
    return predictor


def build_densepose_data_filter(cfg):
    dp_filter = DensePoseDataFilter(cfg)
    return dp_filter


def densepose_inference(densepose_outputs, detections):
    """
    Infer dense pose estimate based on outputs from the DensePose head
    and detections. The estimate for each detection instance is stored in its
    "pred_densepose" attribute.

    Args:
        densepose_outputs (tuple(`torch.Tensor`)): iterable containing 4 elements:
            - s (:obj: `torch.Tensor`): segmentation tensor of size (N, A, H, W),
            - i (:obj: `torch.Tensor`): classification tensor of size (N, C, H, W),
            - u (:obj: `torch.Tensor`): U coordinates for each class of size (N, C, H, W),
            - v (:obj: `torch.Tensor`): V coordinates for each class of size (N, C, H, W),
            where N is the total number of detections in a batch,
                  A is the number of segmentations classes (e.g. 15 for coarse body parts),
                  C is the number of labels (e.g. 25 for fine body parts),
                  W is the resolution along the X axis
                  H is the resolution along the Y axis
        detections (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Instances are modified by this method: "pred_densepose" attribute
            is added to each instance, the attribute contains the corresponding
            DensePoseOutput object.
    """

    # DensePose outputs: segmentation, body part indices, U, V
    s, index_uv, u, v = densepose_outputs
    k = 0
    for detection in detections:
        n_i = len(detection)
        s_i = s[k : k + n_i]
        index_uv_i = index_uv[k : k + n_i]
        u_i = u[k : k + n_i]
        v_i = v[k : k + n_i]
        densepose_output_i = DensePoseOutput(s_i, index_uv_i, u_i, v_i)
        detection.pred_densepose = densepose_output_i
        k += n_i


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


def _grid_sampling_utilities(
    zh, zw, bbox_xywh_est, bbox_xywh_gt, index_gt, x_norm, y_norm, index_bbox
):
    """
    Prepare tensors used in grid sampling.

    Args:
        z_est (:obj: `torch.Tensor`): tensor of size (N,C,H,W) with estimated
            values of Z to be extracted for the points X, Y and channel
            indices I
        bbox_xywh_est (:obj: `torch.Tensor`): tensor of size (N, 4) containing
            estimated bounding boxes in format XYWH
        bbox_xywh_gt (:obj: `torch.Tensor`): tensor of size (N, 4) containing
            matched ground truth bounding boxes in format XYWH
        index_gt (:obj: `torch.Tensor`): tensor of size K with point labels for
            ground truth points
        x_norm (:obj: `torch.Tensor`): tensor of size K with X normalized
            coordinates of ground truth points. Image X coordinates can be
            obtained as X = Xbbox + x_norm * Wbbox / 255
        y_norm (:obj: `torch.Tensor`): tensor of size K with Y normalized
            coordinates of ground truth points. Image Y coordinates can be
            obtained as Y = Ybbox + y_norm * Hbbox / 255
        index_bbox (:obj: `torch.Tensor`): tensor of size K with bounding box
            indices for each ground truth point. The values are thus in
            [0, N-1]

    Returns:
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

    x0_gt, y0_gt, w_gt, h_gt = bbox_xywh_gt[index_bbox].unbind(dim=1)
    x0_est, y0_est, w_est, h_est = bbox_xywh_est[index_bbox].unbind(dim=1)
    x_lo, x_hi, x_w, jx_valid = _linear_interpolation_utilities(
        x_norm, x0_gt, w_gt, x0_est, w_est, zw
    )
    y_lo, y_hi, y_w, jy_valid = _linear_interpolation_utilities(
        y_norm, y0_gt, h_gt, y0_est, h_est, zh
    )
    j_valid = jx_valid * jy_valid

    w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
    w_ylo_xhi = x_w * (1.0 - y_w)
    w_yhi_xlo = (1.0 - x_w) * y_w
    w_yhi_xhi = x_w * y_w

    return j_valid, y_lo, y_hi, x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi


def _extract_at_points_packed(
    z_est,
    index_bbox_valid,
    slice_index_uv,
    y_lo,
    y_hi,
    x_lo,
    x_hi,
    w_ylo_xlo,
    w_ylo_xhi,
    w_yhi_xlo,
    w_yhi_xhi,
):
    """
    Extract ground truth values z_gt for valid point indices and estimated
    values z_est using bilinear interpolation over top-left (y_lo, x_lo),
    top-right (y_lo, x_hi), bottom-left (y_hi, x_lo) and bottom-right
    (y_hi, x_hi) values in z_est with corresponding weights:
    w_ylo_xlo, w_ylo_xhi, w_yhi_xlo and w_yhi_xhi.
    Use slice_index_uv to slice dim=1 in z_est
    """
    z_est_sampled = (
        z_est[index_bbox_valid, slice_index_uv, y_lo, x_lo] * w_ylo_xlo
        + z_est[index_bbox_valid, slice_index_uv, y_lo, x_hi] * w_ylo_xhi
        + z_est[index_bbox_valid, slice_index_uv, y_hi, x_lo] * w_yhi_xlo
        + z_est[index_bbox_valid, slice_index_uv, y_hi, x_hi] * w_yhi_xhi
    )
    return z_est_sampled


def _resample_data(
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


def _extract_single_tensors_from_matches(proposals_with_targets):
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
        i_gt_img, x_norm_img, y_norm_img, u_gt_img, v_gt_img, s_gt_img, bbox_xywh_gt_img, bbox_xywh_est_img, i_bbox_img, i_with_dp_img = _extract_single_tensors_from_matches_one_image(  # noqa
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


class DensePoseLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on

    def __call__(self, proposals_with_gt, densepose_outputs):
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        s, index_uv, u, v = densepose_outputs
        assert u.size(2) == v.size(2)
        assert u.size(3) == v.size(3)
        assert u.size(2) == index_uv.size(2)
        assert u.size(3) == index_uv.size(3)

        with torch.no_grad():
            index_uv_img, i_with_dp, bbox_xywh_est, bbox_xywh_gt, index_gt_all, x_norm, y_norm, u_gt_all, v_gt_all, s_gt, index_bbox = _extract_single_tensors_from_matches(  # noqa
                proposals_with_gt
            )
        n_batch = len(i_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            losses["loss_densepose_U"] = u.sum() * 0
            losses["loss_densepose_V"] = v.sum() * 0
            losses["loss_densepose_I"] = index_uv.sum() * 0
            losses["loss_densepose_S"] = s.sum() * 0
            return losses

        zh = u.size(2)
        zw = u.size(3)

        j_valid, y_lo, y_hi, x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi = _grid_sampling_utilities(  # noqa
            zh, zw, bbox_xywh_est, bbox_xywh_gt, index_gt_all, x_norm, y_norm, index_bbox
        )

        j_valid_fg = j_valid * (index_gt_all > 0)

        u_gt = u_gt_all[j_valid_fg]
        u_est_all = _extract_at_points_packed(
            u[i_with_dp],
            index_bbox,
            index_gt_all,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )
        u_est = u_est_all[j_valid_fg]

        v_gt = v_gt_all[j_valid_fg]
        v_est_all = _extract_at_points_packed(
            v[i_with_dp],
            index_bbox,
            index_gt_all,
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo,
            w_ylo_xhi,
            w_yhi_xlo,
            w_yhi_xhi,
        )
        v_est = v_est_all[j_valid_fg]

        index_uv_gt = index_gt_all[j_valid]
        index_uv_est_all = _extract_at_points_packed(
            index_uv[i_with_dp],
            index_bbox,
            slice(None),
            y_lo,
            y_hi,
            x_lo,
            x_hi,
            w_ylo_xlo[:, None],
            w_ylo_xhi[:, None],
            w_yhi_xlo[:, None],
            w_yhi_xhi[:, None],
        )
        index_uv_est = index_uv_est_all[j_valid, :]

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        s_est = s[i_with_dp]
        with torch.no_grad():
            s_gt = _resample_data(
                s_gt.unsqueeze(1),
                bbox_xywh_gt,
                bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)

        # add point-based losses:
        u_loss = F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points
        losses["loss_densepose_U"] = u_loss
        v_loss = F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points
        losses["loss_densepose_V"] = v_loss
        index_uv_loss = F.cross_entropy(index_uv_est, index_uv_gt.long()) * self.w_part
        losses["loss_densepose_I"] = index_uv_loss

        if self.n_segm_chan == 2:
            s_gt = s_gt > 0
        s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
        losses["loss_densepose_S"] = s_loss
        return losses


def build_densepose_losses(cfg):
    losses = DensePoseLosses(cfg)
    return losses
