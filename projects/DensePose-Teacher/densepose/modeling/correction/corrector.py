# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from abc import ABC
from dataclasses import dataclass
import logging
from typing import Dict, Optional, List, Any, Union, Tuple
from detectron2.layers.wrappers import BatchNorm2d

from detectron2.structures import Instances, BoxMode
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d, interpolate, ConvTranspose2d, ASPP, get_norm
from ..utils import initialize_module_params
from detectron2.utils.file_io import PathManager

from densepose import DensePoseDataRelative


def _linear_interpolation_utilities(v_norm, v0_src, size_src, size_z):
    v = v0_src + v_norm * size_src / 256
    v_grid = v * size_z / size_src
    v_lo = v.floor().long().clamp(min=0, max=size_z - 1)
    v_hi = (v_lo + 1).clamp(max=size_z - 1)
    v_grid = torch.min(v_hi.float(), v_grid)
    v_w = v_grid - v_lo.float()

    return v_lo, v_hi, v_w


@dataclass
class PackedAnnotations:
    coarse_segm_gt: Optional[torch.Tensor]
    fine_segm_labels_gt: torch.Tensor
    x_gt: torch.Tensor
    y_gt: torch.Tensor
    u_gt: torch.Tensor
    v_gt: torch.Tensor
    bbox_xywh_gt: torch.Tensor
    point_bbox_with_dp_indices: torch.Tensor
    point_bbox_indices: torch.Tensor
    bbox_indices: torch.Tensor


class Accumulator(ABC):
    def __init__(self):
        self.s_gt = []
        self.i_gt = []
        self.x_gt = []
        self.y_gt = []
        self.u_gt = []
        self.v_gt = []
        self.bbox_xywh_gt = []
        self.point_bbox_with_dp_indices = []
        self.point_bbox_indices = []
        self.bbox_indices = []
        self.nxt_bbox_with_dp_index = 0
        self.nxt_bbox_index = 0

    def accumulate(self, instances_one_image: Instances):
        boxes_xywh_gt = BoxMode.convert(
            instances_one_image.gt_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )
        for box_xywh_gt, dp_gt in zip(boxes_xywh_gt, instances_one_image.gt_densepose):
            if (dp_gt is not None) and (len(dp_gt.x) > 0):
                self._do_accumulate(box_xywh_gt, dp_gt)
            self.nxt_bbox_index += 1

    def _do_accumulate(self, box_xywh_gt: torch.Tensor, dp_gt: DensePoseDataRelative):
        self.i_gt.append(dp_gt.i)
        self.x_gt.append(dp_gt.x)
        self.y_gt.append(dp_gt.y)
        self.u_gt.append(dp_gt.u)
        self.v_gt.append(dp_gt.v)
        if hasattr(dp_gt, "segm"):
            self.s_gt.append(dp_gt.segm.unsqueeze(0))
        self.bbox_xywh_gt.append(box_xywh_gt.view(-1, 4))
        self.point_bbox_with_dp_indices.append(
            torch.full_like(dp_gt.i, self.nxt_bbox_with_dp_index)
        )
        self.point_bbox_indices.append(torch.full_like(dp_gt.i, self.nxt_bbox_index))
        self.bbox_indices.append(self.nxt_bbox_index)
        self.nxt_bbox_with_dp_index += 1

    def pack(self) -> Optional[PackedAnnotations]:
        if not len(self.i_gt):
            return None
        return PackedAnnotations(
            fine_segm_labels_gt=torch.cat(self.i_gt, 0).long(),
            x_gt=torch.cat(self.x_gt, 0),
            y_gt=torch.cat(self.y_gt, 0),
            u_gt=torch.cat(self.u_gt, 0),
            v_gt=torch.cat(self.v_gt, 0),
            coarse_segm_gt=torch.cat(self.s_gt, 0)
            if len(self.s_gt) == len(self.bbox_xywh_gt)
            else None,
            bbox_xywh_gt=torch.cat(self.bbox_xywh_gt, 0),
            point_bbox_with_dp_indices=torch.cat(self.point_bbox_with_dp_indices, 0).long(),
            point_bbox_indices=torch.cat(self.point_bbox_indices, 0).long(),
            bbox_indices=torch.as_tensor(
                self.bbox_indices, dtype=torch.long, device=self.x_gt[0].device
            ).long(),
        )


class InterpolationHelper:
    def __init__(
        self,
        packed_annotations: Any,
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
    ):
        zh, zw = densepose_outputs_size_hw
        x0_gt, y0_gt, w_gt, h_gt = packed_annotations.bbox_xywh_gt[
            packed_annotations.point_bbox_with_dp_indices
        ].unbind(dim=1)
        x_lo, x_hi, x_w = _linear_interpolation_utilities(packed_annotations.x_gt, x0_gt, w_gt, zw)
        y_lo, y_hi, y_w = _linear_interpolation_utilities(packed_annotations.y_gt, y0_gt, h_gt, zh)

        w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
        w_ylo_xhi = x_w * (1.0 - y_w)
        w_yhi_xlo = (1.0 - x_w) * y_w
        w_yhi_xhi = x_w * y_w

        return InterpolationHelper(
            packed_annotations,
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


class Corrector(nn.Module):

    DEFAULT_MODEL_CHECKPOINT_PREFIX = "roi_heads.corrector."

    def __init__(self, cfg: CfgNode):
        """
        Initialize mesh correctors. An corrector for mesh `i` is stored in a submodule
        "corrector_{i}".

        Args:
            cfg (CfgNode): configuration options
        """
        super(Corrector, self).__init__()
        hidden_dim = cfg.MODEL.SEMI.COR.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.SEMI.COR.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.SEMI.COR.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_pred_channels = (cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1) * 3 + cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        n_channels = n_pred_channels + 256 + 1 # + cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # self.upsample = ConvTranspose2d(n_pool_channels, n_pred_channels, 4, stride=2, padding=1)
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        self.predictor = CorrectorPredictor(cfg, self.n_out_channels)
        initialize_module_params(self)
        self.non_local = NonLocalBlock(in_channels=n_channels)
        # self.aspp = ASPP(in_channels=n_channels, out_channels=n_channels, dilations=[6, 12, 18],
        #                     norm="BN", activation=F.relu)

        self.w_segm = cfg.MODEL.SEMI.COR.SEGM_WEIGHTS

        self.w_points = cfg.MODEL.SEMI.COR.POINTS_WEIGHTS
        self.patch_channels = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.correct_warm_iter = cfg.MODEL.SEMI.COR.WARM_ITER
        if self.correct_warm_iter == 0:
            self.correct_warm_iter = cfg.SOLVER.MAX_ITER

        # self.symm_parts_index = [0, 2, 1, 3, 4, 5, 6, 9, 7, 10, 8, 13, 11, 14, 12, 17, 15, 18, 16, 21, 19, 22, 20, 24, 23]

        logger = logging.getLogger(__name__)
        logger.info(f"Adding Corrector ...")
        if cfg.MODEL.WEIGHTS != "":
            self.load_from_model_checkpoint(cfg.MODEL.WEIGHTS)

    # def forward(self, instances: List[Instances]):
    # # def forward(self, predictor_output, pool_feats):
    #     pool_feats = torch.cat([x.pool_feat for x in instances], dim=0)
    #     # pred_densepose_coarse_segm = torch.cat([x.pred_densepose.coarse_segm for x in instances], dim=0)
    #     pred_densepose_segm = torch.cat([x.pred_densepose.fine_segm for x in instances], dim=0)
    #     pred_densepose_u = torch.cat([x.pred_densepose.u for x in instances], dim=0)
    #     pred_densepose_v = torch.cat([x.pred_densepose.v for x in instances], dim=0)
    #     # pred_densepose_segm = predictor_output.fine_segm
    #     # pred_densepose_u = predictor_output.u
    #     # pred_densepose_v = predictor_output.v
    #     # process input features and prediction
    #     # pred_densepose_segm = F.softmax(pred_densepose_segm, dim=1)
    #     # pred_densepose_u = (pred_densepose_u * 255.0).clamp(0, 255.0) / 255.0
    #     # pred_densepose_v = (pred_densepose_v * 255.0).clamp(0, 255.0) / 255.0

    #     # pred_densepose_coarse_segm = F.interpolate(pred_densepose_coarse_segm, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False)
    #     # pred_densepose_segm = F.interpolate(pred_densepose_segm, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False)
    #     # pred_densepose_u = F.interpolate(pred_densepose_u, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False)
    #     # pred_densepose_v = F.interpolate(pred_densepose_v, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False)

    #     # pred_densepose_segm = down_sampling(pred_densepose_segm, pool_feats.size(-2), pool_feats.size(-1))
    #     # pred_densepose_u = down_sampling(pred_densepose_u, pool_feats.size(-2), pool_feats.size(-1))
    #     # pred_densepose_v = down_sampling(pred_densepose_v, pool_feats.size(-1), pool_feats.size(-1))
        
    #     # concat the pred and output
    #     # pool_feats = self.upsample(pool_feats)
    #     # pool_feats = interpolate(pool_feats, scale_factor=2, mode='bilinear', align_corners=False)
    #     output = torch.cat((
    #         pool_feats,
    #         # F.interpolate(pred_densepose_coarse_segm, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False),
    #         F.interpolate(pred_densepose_segm, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False),
    #         F.interpolate(pred_densepose_u, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False),
    #         F.interpolate(pred_densepose_v, size=pool_feats.shape[-2:], mode='bilinear', align_corners=False)
    #     ), dim=1)
    #     for i in range(self.n_stacked_convs):
    #         layer_name = self._get_layer_name(i)
    #         output = getattr(self, layer_name)(output)
    #         output = F.relu(output)
    #         if (i + 1) == self.n_stacked_convs // 2:
    #             output = self.non_local(output)
    #     output = self.predictor(output)

    #     # return output
    #     if self.training:
    #         # get packed gt value
    #         accumulator = Accumulator()
    #         for instance in instances:
    #             accumulator.accumulate(instance)
    #         packed_annotations = accumulator.pack()
    #         if packed_annotations is None:
    #             segm_loss = self.fake_segm_loss(output)
    #             points_loss = self.fake_points_loss(output)
    #         else:
    #             h, w = output.u.shape[-2:]
    #             interpolator = InterpolationHelper.from_matches(
    #                 packed_annotations,
    #                 (h, w),
    #             )
    #             segm_loss = self.segm_loss(
    #                 output,
    #                 packed_annotations,
    #                 interpolator,
    #                 pred_densepose_segm,
    #                 torch.cat([x.pred_densepose.coarse_segm for x in instances], dim=0)
    #             )

    #             points_loss = self.points_loss(
    #                 output,
    #                 packed_annotations,
    #                 interpolator,
    #                 pred_densepose_u,
    #                 pred_densepose_v,
    #             )

    #         return output, {**segm_loss, **points_loss}

    #     return output, None

    def forward(self, features_dp, predictor_outputs, shuffle=None):
        with torch.no_grad():
            fine_segm = F.interpolate(predictor_outputs.fine_segm, size=features_dp.shape[-2:], mode='bilinear', align_corners=False)
            coarse_segm = F.interpolate(predictor_outputs.coarse_segm, size=features_dp.shape[-2:], mode='bilinear', align_corners=False)
            p = F.softmax(fine_segm, dim=1)
            coarse_segm = F.softmax(coarse_segm, dim=1)

            u = F.interpolate(predictor_outputs.u, size=features_dp.shape[-2:], mode='bilinear', align_corners=False).clamp(0., 1.)
            v = F.interpolate(predictor_outputs.v, size=features_dp.shape[-2:], mode='bilinear', align_corners=False).clamp(0., 1.)

            fine_segm_entropy = torch.sum(-p * F.log_softmax(fine_segm, dim=1), dim=1).unsqueeze(1)
            # coarse_segm_entropy = torch.sum(-coarse_segm * F.log_softmax(coarse_segm, dim=1), dim=1).unsqueeze(1)

            features_input = features_dp.detach()

        output = torch.cat((features_input, coarse_segm, p, fine_segm_entropy, u, v), dim=1)
        # output = torch.cat((features_input, p, fine_segm_entropy, u, v), dim=1)

        # if self.training and shuffle is not None:
        #     output_2 = torch.cat((features_input, p[:, shuffle, :, :], fine_segm_entropy, u, v), dim=1)
        #     output = torch.cat((output, output_2), dim=0)

        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            output = getattr(self, layer_name)(output)
            output = F.relu(output)
            if (i + 1) == self.n_stacked_convs // 2:
                # output = self.aspp(output)
                output = self.non_local(output)
        return self.predictor(output)

    def interp(self, tensor_nchw):
        return interpolate(
            tensor_nchw, scale_factor=2, mode="bilinear", align_corners=False
        )

    # def correct(self, corrections, teacher_outputs: List[Instances], cur_iter=None):
    def correct(self, corrections, teacher_outputs, cur_iter=None):
        # with torch.no_grad():
        #     crt_segm = F.softmax(corrections.fine_segm, dim=1)
        #     pos_idx = (torch.argmax(crt_segm, dim=1) == self.patch_channels).unsqueeze(1)
        #     k = 0
        #     for output in teacher_outputs:
        #         if cur_iter is None:
        #             correct_rate = 1 - (1 + self.correct_warm_iter) / (50000 + self.correct_warm_iter)
        #         else:
        #             correct_rate = 1 - (1 + self.correct_warm_iter) / (cur_iter + 1 + self.correct_warm_iter)
        #         # else:
        #         #     correct_rate = 1 - (1 + self.correct_warm_iter) / (120000 + self.correct_warm_iter)
        #         n_i = len(output)
        #         # correct pseudo segm
        #         cur_crt_segm = crt_segm[k:k + n_i, :-1]
        #         cur_pos_idx = pos_idx[k:k + n_i].expand(-1, self.patch_channels, -1, -1)
        #         # output.pred_densepose.fine_segm = self.interp(output.pred_densepose.fine_segm)
        #         output.pred_densepose.fine_segm[~cur_pos_idx] = output.pred_densepose.fine_segm[~cur_pos_idx] * (1 - correct_rate) + \
        #                                                         cur_crt_segm[~cur_pos_idx] * correct_rate
        #         # output.pred_densepose.fine_segm[~cur_pos_idx] = cur_crt_segm[~cur_pos_idx]

        #         # correct pseudo uv coordinates
        #         u_correction = ((corrections.u[k:k + n_i] * 255.0).clamp(-255.0, 255.0) / 255.0 * 0.05) * correct_rate
        #         v_correction = ((corrections.v[k:k + n_i] * 255.0).clamp(-255.0, 255.0) / 255.0 * 0.05) * correct_rate
        #         # u_correction = (corrections.u[k: k + n_i].unsqueeze(1).clamp(0, 0.07)) * correct_rate
        #         # v_correction = (corrections.v[k, k + n_i].unsqueeze(1).clamp(0, 0.07)) * correct_rate
        #         output.pred_densepose.u = (u_correction + output.pred_densepose.u).clamp(0., 1.)
        #         output.pred_densepose.v = (v_correction + output.pred_densepose.v).clamp(0., 1.)

        #         k += n_i
        # if cur_iter is None:
        #     correct_rate = 1
        # else:
        #     correct_rate = 1 - (1 + self.correct_warm_iter) / (cur_iter + 1 + self.correct_warm_iter)
        with torch.no_grad():
            # crt_segm = F.softmax(corrections.fine_segm, dim=1).permute(0, 2, 3, 1)
            # pos_idx = (torch.argmax(crt_segm, dim=1) == self.patch_channels).unsqueeze(1)
            # pos_idx = (torch.argmax(crt_segm, dim=3) > 0)

            # if cur_iter is None:
            #     correct_rate = 1
            # else:
            #     correct_rate = 1 - (1 + self.correct_warm_iter) / (cur_iter + 1 + self.correct_warm_iter)

            # u_corrections = corrections.u.clamp(-1, 1) # * correct_rate
            # v_corrections = corrections.v.clamp(-1, 1) # * correct_rate
            # teacher_outputs.u = (u_corrections + teacher_outputs.u.clamp(0., 1.)).clamp(0., 1.)
            # teacher_outputs.v = (v_corrections + teacher_outputs.v.clamp(0., 1.)).clamp(0., 1.)

            # teacher_outputs.fine_segm[~pos_idx] = (teacher_outputs.fine_segm[~pos_idx] * (1 - correct_rate) + \
            #                                     corrections.fine_segm[:, :-1, :, :][~pos_idx]) * correct_rate
            # teacher_outputs.fine_segm = F.softmax(teacher_outputs.fine_segm, dim=1)
            # set pseudo fine_segm
            # teacher_outputs.fine_segm = teacher_outputs.fine_segm * torch.repeat_interleave(F.softplus(corrections.fine_segm) + 0.01, teacher_outputs.fine_segm.shape[1], dim=1)
            # teacher_outputs.fine_segm[~pos_idx] = 0.
            teacher_outputs.pos_map = corrections.fine_segm

    # def segm_loss(self, correction, packed_annotations, interpolator, fine_segm, coarse_segm):
    #     #segm loss - first construct gt for corrector
    #     segm_pred = interpolator.extract_at_points(
    #         fine_segm,
    #         slice_fine_segm=slice(None),
    #         w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
    #         w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
    #         w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
    #         w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
    #     ).argmax(dim=1).long()
    #     segm_gt = packed_annotations.fine_segm_labels_gt
    #     segm_crt_gt = torch.ones_like(segm_gt) * self.patch_channels
    #     index = segm_pred != segm_gt
    #     segm_crt_gt[index] = segm_gt[index]
    #
    #     # get correction pred
    #     segm_crt_est = interpolator.extract_at_points(
    #         correction.fine_segm,
    #         slice_fine_segm=slice(None),
    #         w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
    #         w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
    #         w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
    #         w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
    #     )

        # coarse_segm_pred = torch.argmax(coarse_segm[packed_annotations.bbox_indices], dim=1)
        # coarse_segm_gt = F.interpolate(packed_annotations.coarse_segm_gt.unsqueeze(1), size=coarse_segm_pred.shape[-2:],
        #                                mode='nearest', align_corners=True) > 0
        # coarse_segm_gt = down_sampling(packed_annotations.coarse_segm_gt.unsqueeze(1), coarse_segm_pred.shape[-2],
        #                                coarse_segm_pred.shape[-1]).squeeze(1)

        # coarse_segm_crt_gt = coarse_segm_gt == coarse_segm_pred

        # loss = F.cross_entropy(segm_crt_est, segm_crt_gt.long(), reduction='none')
        # loss[index] = loss[index] * 2
        # return {
        #     "loss_correction_I": F.cross_entropy(segm_crt_est, segm_crt_gt.long()) * self.w_segm,
            # "loss_correction_S": F.cross_entropy(correction.coarse_segm[packed_annotations.bbox_indices],
            #                                      coarse_segm_crt_gt.long()) * self.w_segm * 5
        # }

    # def points_loss(self, correction, packed_annotations, interpolator, u, v):
    #     # uv loss - first construct gt for corrector
    #     u_pred = interpolator.extract_at_points(u)
    #     v_pred = interpolator.extract_at_points(v)
    #     u_crt_gt = packed_annotations.u_gt - u_pred
    #     v_crt_gt = packed_annotations.v_gt - v_pred
    #
    #     u_crt_est = interpolator.extract_at_points(correction.u)
    #     v_crt_est = interpolator.extract_at_points(correction.v)
    #
    #     return {
    #         "loss_correction_U": F.smooth_l1_loss(u_crt_est, u_crt_gt, reduction='sum') * self.w_points,
    #         "loss_correction_V": F.smooth_l1_loss(v_crt_est, v_crt_gt, reduction='sum') * self.w_points,
    #     }

    # def fake_segm_loss(self, correction):
    #     return {
    #         "loss_correction_I": correction.fine_segm.sum() * 0,
    #         "loss_correction_S": correction.coarse_segm.sum() * 0
    #     }
    #
    # def fake_points_loss(self, correction):
    #     return {
    #         "loss_correction_U": correction.u.sum() * 0,
    #         "loss_correction_V": correction.v.sum() * 0,
    #     }

    def _get_layer_name(self, i: int):
        layer_name = "corrector_conv_fcn{}".format(i + 1)
        return layer_name

    def load_from_model_checkpoint(self, fpath: str, prefix: Optional[str] = None):
        import numpy as np

        if prefix is None:
            prefix = Corrector.DEFAULT_MODEL_CHECKPOINT_PREFIX
        if fpath.endswith(".pkl"):
            return
        with PathManager.open(fpath, "rb") as hFile:
                state_dict = torch.load(hFile, map_location=torch.device("cpu"))
        if state_dict is not None and "model" in state_dict:
            state_dict_local = {}
            for key in state_dict["model"]:
                if key.startswith(prefix):
                    v_key = state_dict["model"][key]
                    if isinstance(v_key, np.ndarray):
                        v_key = torch.from_numpy(v_key)
                    state_dict_local[key[len(prefix) :]] = v_key
            self.load_state_dict(state_dict_local, strict=False)

def down_sampling(z, wout, hout, mode: str='nearest', padding_mode:str='zeros'):
    n = z.shape[0]
    grid_w = torch.arange(wout, device=z.device, dtype=torch.float) / wout
    grid_h = torch.arange(hout, device=z.device, dtype=torch.float) / hout
    grid_w = grid_w[None, None, :].expand(n, hout, wout) * 2 - 1
    grid_h = grid_h[None, :, None].expand(n, hout, wout) * 2 - 1
    grid = torch.stack((grid_w, grid_h), dim=3)
    return F.grid_sample(z, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    # def load_from_model_checkpoint(self, fpath: str, prefix: Optional[str] = None):
    #     if prefix is None:
    #         prefix = Corrector.DEFAULT_MODEL_CHECKPOINT_PREFIX
    #     state_dict = None
    #     if fpath.endswith(".pkl"):
    #         with PathManager.open(fpath, "rb") as hFile:
    #             state_dict = pickle.load(hFile, encoding="latin1")  # pyre-ignore[6]
    #     else:
    #         with PathManager.open(fpath, "rb") as hFile:
    #             state_dict = torch.load(hFile, map_location=torch.device("cpu"))
    #     if state_dict is not None and "model" in state_dict:
    #         state_dict_local = {}
    #         for key in state_dict["model"]:
    #             if key.startswith(prefix):
    #                 v_key = state_dict["model"][key]
    #                 if isinstance(v_key, np.ndarray):
    #                     v_key = torch.from_numpy(v_key)
    #                 state_dict_local[key[len(prefix) :]] = v_key
    #         # non-strict loading to finetune on different meshes
    #         # pyre-fixme[6]: Expected `OrderedDict[typing.Any, typing.Any]` for 1st
    #         #  param but got `Dict[typing.Any, typing.Any]`.
    #         self.load_state_dict(state_dict_local, strict=False)


class CorrectorPredictor(nn.Module):
    def __init__(self, cfg: CfgNode, input_channels: int):
        super().__init__()
        dim_in = input_channels
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        # kernel_size = cfg.MODEL.SEMI.COR.CONV_HEAD_KERNEL
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL

        # self.segm_correction = Conv2d(
        #     dim_in, dim_out_patches + 1, kernel_size, stride=1, padding=kernel_size//2
        # )
        #
        # self.u_correction = Conv2d(
        #     dim_in, dim_out_patches, kernel_size, stride=1, padding=kernel_size//2
        # )
        #
        # self.v_correction = Conv2d(
        #     dim_in, dim_out_patches, kernel_size, stride=1, padding=kernel_size//2
        # )
        self.ann_index_correction = ConvTranspose2d(
            dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )

        self.segm_correction = ConvTranspose2d(
            dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )

        # self.u_correction = ConvTranspose2d(
        #     dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )

        # self.v_correction = ConvTranspose2d(
        #     dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        # )
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE
        initialize_module_params(self)

    def interp2d(self, tensor_nchw: torch.Tensor):
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

    def forward(self, corrector_output: torch.Tensor):
        return CorrectorPredictorOutput(
            coarse_segm=self.interp2d(self.ann_index_correction(corrector_output)),
            fine_segm=self.interp2d(self.segm_correction(corrector_output)),
            # u=self.interp2d(self.u_correction(corrector_output)),
            # v=self.interp2d(self.v_correction(corrector_output)),
        )  


@dataclass
class CorrectorPredictorOutput:
    coarse_segm: torch.Tensor
    fine_segm: torch.Tensor
    # u: torch.Tensor
    # v: torch.Tensor

    def __len__(self):
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ):
        if isinstance(item, int):
            return CorrectorPredictorOutput(
                coarse_segm=self.coarse_segm.unsqueeze(0),
                fine_segm=self.fine_segm[item].unsqueeze(0),
                # u=self.u[item].unsqueeze(0),
                # v=self.v[item].unsqueeze(0),
            )
        else:
            return CorrectorPredictorOutput(
                coarse_segm=self.coarse_segm[item],
                fine_segm=self.fine_segm[item],
                # u=self.u[item],
                # v=self.v[item],
            )

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        coarse_segm = self.coarse_segm.to(device)
        fine_segm = self.fine_segm.to(device)
        # u = self.u.to(device)
        # v = self.v.to(device)
        return CorrectorPredictorOutput(coarse_segm=coarse_segm, fine_segm=fine_segm)#, u=u, v=v)
        # return CorrectorPredictorOutput(fine_segm=fine_segm, u=u, v=v)

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True) -> None:
        super().__init__()
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        
        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)
        
        if self.mode == 'embedded' or self.mode == 'dot' or self.mode == 'concatenate':
            self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == 'concatenate':
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size = x.shape[0]

        if batch_size == 0:
            return x

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == 'gaussian':
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'embedded' or self.mode == 'dot':
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == 'concatenate':
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.sze(0), f.size(2), f.size(3))

        if self.mode == 'gaussian' or self.mode == 'embedded':
            f_div_C = F.softmax(f, dim=1)
        elif self.mode == 'dot' or self.mode == 'concatenate':
            N = f.size(-1)
            f_div_c = f / N

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        z = W_y + x

        return z
