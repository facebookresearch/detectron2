# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import torch
from torch.nn import functional as F
import os
import json

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import BilinearInterpolationHelper, LossDict, SingleTensorsHelper, resample_data

import scipy.spatial.distance as ssd
from scipy.io import loadmat
import numpy as np
import pickle


@dataclass
class DataForMaskLoss:
    """
    Contains mask GT and estimated data for proposals from multiple images:
    """

    # tensor of size (K, H, W) containing GT labels
    masks_gt: Optional[torch.Tensor] = None
    # tensor of size (K, C, H, W) containing estimated scores
    masks_est: Optional[torch.Tensor] = None


def extract_data_for_mask_loss_from_matches(
    proposals_targets: Iterable[Instances], estimated_segm: torch.Tensor
) -> DataForMaskLoss:
    """
    Extract data for mask loss from instances that contain matched GT and
    estimated bounding boxes.
    Args:
        proposals_targets: Iterable[Instances]
            matched GT and estimated results, each item in the iterable
            corresponds to data in 1 image
        estimated_segm: tensor(K, C, S, S) of float - raw unnormalized
            segmentation scores, here S is the size to which GT masks are
            to be resized
    Return:
        masks_est: tensor(K, C, S, S) of float - class scores
        masks_gt: tensor(K, S, S) of int64 - labels
    """
    data = DataForMaskLoss()
    masks_gt = []
    offset = 0
    assert estimated_segm.shape[2] == estimated_segm.shape[3], (
        f"Expected estimated segmentation to have a square shape, "
        f"but the actual shape is {estimated_segm.shape[2:]}"
    )
    mask_size = estimated_segm.shape[2]
    num_proposals = sum(inst.proposal_boxes.tensor.size(0) for inst in proposals_targets)
    num_estimated = estimated_segm.shape[0]
    assert (
        num_proposals == num_estimated
    ), "The number of proposals {} must be equal to the number of estimates {}".format(
        num_proposals, num_estimated
    )

    for proposals_targets_per_image in proposals_targets:
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        gt_masks_per_image = proposals_targets_per_image.gt_masks.crop_and_resize(
            proposals_targets_per_image.proposal_boxes.tensor, mask_size
        ).to(device=estimated_segm.device)
        masks_gt.append(gt_masks_per_image)
        offset += n_i
    if masks_gt:
        data.masks_est = estimated_segm
        data.masks_gt = torch.cat(masks_gt, dim=0)
    return data


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseMeshChartLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U coordinates, tensor of shape [N, C, S, S]
     * V coordinates, tensor of shape [N, C, S, S]
     * fine segmentation estimates, tensor of shape [N, C, S, S] with raw unnormalized
       scores for each fine segmentation label at each location
     * coarse segmentation estimates, tensor of shape [N, D, S, S] with raw unnormalized
       scores for each coarse segmentation label at each location
    where N is the number of detections, C is the number of fine segmentation
    labels, S is the estimate size ( = width = height) and D is the number of
    coarse segmentation channels.

    The losses are:
    * regression (smooth L1) loss for U and V coordinates
    * cross entropy loss for fine (I) and coarse (S) segmentations
    Each loss has an associated weight
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize chart-based loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        self.n_i_chan     = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.eql = (cfg.MODEL.ROI_DENSEPOSE_HEAD.SEGLOSS_TYPE=="eql")
        self.eql_lambda = cfg.MODEL.ROI_DENSEPOSE_HEAD.EQL_LAMBDA
        self.gamma = 0.9
        self.freq = cfg.MODEL.ROI_DENSEPOSE_HEAD.CLASS_FREQ
        self.threshold_func_type = cfg.MODEL.ROI_DENSEPOSE_HEAD.THRESHOLD_FUNC_TYPE
        # mesh loss
        self.mesh_uv_loss = cfg.MODEL.ROI_DENSEPOSE_HEAD.MESH_UVLOSS
        self.mesh_with_classify = cfg.MODEL.ROI_DENSEPOSE_HEAD.MESH_WITH_CLASSIFY
        self.w_mesh = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_MESH_WEIGHT
        smpl_subdiv_fpath = '/home/sunjunyao/code/Model/pretrained_model/SMPL_subdiv.mat'
        pdist_transform_fpath = '/home/sunjunyao/code/Model/pretrained_model/SMPL_SUBDIV_TRANSFORM.mat'
        pdist_matrix_fpath = '/home/sunjunyao/code/Model/pretrained_model/Pdist_matrix.pkl'
        SMPL_subdiv = loadmat(smpl_subdiv_fpath)
        # self.Vertex = torch.from_numpy(SMPL_subdiv["vertex"].astype(float)).cuda()
        PDIST_transform = loadmat(pdist_transform_fpath)["index"].astype(np.int32)
        self.PDIST_transform = torch.from_numpy(PDIST_transform.squeeze()).cuda()
        UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
        ClosestVertInds = np.arange(UV.shape[1]) + 1
        self.Part_UVs = []
        self.Part_ClosestVertInds = []
        for i in range(24):
            self.Part_UVs.append(UV[:, SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)])
            self.Part_ClosestVertInds.append(
                ClosestVertInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]
            )
        with open(pdist_matrix_fpath, "rb") as hFile:
            arrays = pickle.load(hFile, encoding="latin1")
        # self.Pdist_matrix = torch.from_numpy(arrays["Pdist_matrix"]).cuda()
        self.Pdist_matrix = arrays["Pdist_matrix"]
        self.Part_ids = np.array(SMPL_subdiv["Part_ID_subdiv"].squeeze())

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any
    ) -> LossDict:
        """
        Produce chart-based DensePose losses

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
                * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
                * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
                * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
            where N is the number of detections, C is the number of fine segmentation
            labels, S is the estimate size ( = width = height) and D is the number of
            coarse segmentation channels.

        Return:
            (dict: str -> tensor): dict of losses with the following entries:
                * loss_densepose_I: fine segmentation loss (cross-entropy)
                * loss_densepose_S: coarse segmentation loss (cross-entropy)
                * loss_densepose_U: loss for U coordinates (smooth L1)
                * loss_densepose_V: loss for V coordinates (smooth L1)
        """
        if not self.segm_trained_by_masks:
            return self.produce_densepose_losses(proposals_with_gt, densepose_predictor_outputs)
        else:
            losses_densepose = self.produce_densepose_losses(
                proposals_with_gt, densepose_predictor_outputs
            )
            losses_mask = self.produce_mask_losses(proposals_with_gt, densepose_predictor_outputs)
            return {**losses_densepose, **losses_mask}

    def produce_fake_mask_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake coarse segmentation loss used when no suitable ground truth data
        was found in a batch. The loss has a value 0 and is primarily used to
        construct the computation graph, so that `DistributedDataParallel`
        has similar graphs on all GPUs and can perform reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have `coarse_segm`
                attribute
        Return:
            dict: str -> tensor: dict of losses with a single entry
                `loss_densepose_S` with 0 value
        """
        return {"loss_densepose_S": densepose_predictor_outputs.coarse_segm.sum() * 0}

    def produce_mask_losses(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any
    ) -> LossDict:
        """
        Computes coarse segmentation loss as cross-entropy for raw unnormalized
        scores given ground truth labels.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attribute:
                * coarse_segm (tensor of shape [N, D, S, S]): coarse segmentation estimates
                    as raw unnormalized scores
            where N is the number of detections, S is the estimate size ( = width = height) and
            D is the number of coarse segmentation channels.
        Return:
            dict: str -> tensor: dict of losses with a single entry:
            * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                segmentation given ground truth labels
        """
        if not len(proposals_with_gt):
            return self.produce_fake_mask_losses(densepose_predictor_outputs)
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        with torch.no_grad():
            mask_loss_data = extract_data_for_mask_loss_from_matches(
                proposals_with_gt, densepose_predictor_outputs.coarse_segm
            )
        if (mask_loss_data.masks_gt is None) or (mask_loss_data.masks_est is None):
            return self.produce_fake_mask_losses(densepose_predictor_outputs)
        return {
            "loss_densepose_S": F.cross_entropy(
                mask_loss_data.masks_est, mask_loss_data.masks_gt.long()
            )
            * self.w_segm
        }

    def produce_fake_densepose_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine segmentation and U/V coordinates. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
        """
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for U/V coordinates. These are used when no suitable ground
        truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
        """
        if self.mesh_uv_loss:
            return {
                "loss_densepose_U": densepose_predictor_outputs.u.sum() * 0,
                "loss_densepose_V": densepose_predictor_outputs.v.sum() * 0,
                "loss_densepose_mesh": densepose_predictor_outputs.u.sum() * 0,
            }
        return {
            "loss_densepose_mesh": densepose_predictor_outputs.u.sum() * 0,
        }

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine / coarse segmentation. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
        """
        losses = {"loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0}
        if not self.segm_trained_by_masks:
            losses["loss_densepose_S"] = densepose_predictor_outputs.coarse_segm.sum() * 0
        return losses

    def produce_densepose_losses(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any
    ) -> LossDict:
        """
        Losses for segmentation and U/V coordinates computed as cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points and smooth L1 loss for U and V coordinate estimates at
        annotated points.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        densepose_outputs_size = densepose_predictor_outputs.u.size()

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        tensors_helper = SingleTensorsHelper(proposals_with_gt)
        n_batch = len(tensors_helper.index_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        interpolator = BilinearInterpolationHelper.from_matches(
            tensors_helper, densepose_outputs_size
        )

        j_valid_fg = interpolator.j_valid * (tensors_helper.fine_segm_labels_gt > 0)

        losses_uv = self.produce_densepose_losses_uv(
            proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg
        )

        losses_segm = self.produce_densepose_losses_segm(
            proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg
        )

        return {**losses_uv, **losses_segm}

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        tensors_helper: SingleTensorsHelper,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Compute losses for U/V coordinates: smooth L1 loss between
        estimated coordinates and the ground truth.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
        """
        u_gt = tensors_helper.u_gt[j_valid_fg]
        v_gt = tensors_helper.v_gt[j_valid_fg]
        i_gt = tensors_helper.fine_segm_labels_gt[j_valid_fg] # J
        i_est = torch.argmax(interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm[tensors_helper.index_with_dp],
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        ), axis=1)
        # slice_fine_segm = i_est.detach()
        u_est = interpolator.extract_at_points(
            densepose_predictor_outputs.u[tensors_helper.index_with_dp]
        )[j_valid_fg]
        v_est = interpolator.extract_at_points(
            densepose_predictor_outputs.v[tensors_helper.index_with_dp]
        )[j_valid_fg]
        i_est = i_est[j_valid_fg]
        assert len(i_est) == len(u_est)
        if self.mesh_with_classify:
            cVerts, cVertsGT, cVertsProb, cVertsGTProb = self.findAllClosestVerts(u_gt, v_gt, i_gt, u_est, v_est, i_est)
            cVertsProbM = torch.where(cVertsProb>1., torch.tensor(0, dtype=torch.float).cuda(), cVertsProb)      
            # dist = torch.tensor([self.getDistances(cVertsGT, cVerts[:,r]) for r in range(2)]).transpose(1,0).cuda()
            dist = torch.from_numpy(self.getDistances(cVertsGT, cVerts)).cuda()
            loss_mesh = - torch.log(torch.true_divide(torch.sum(torch.exp(-dist)*cVertsProbM), dist.shape[0]))
            # loss_mesh_dist = torch.sum(torch.where(torch.abs(dist)<1., 0.5 * (dist ** 2), torch.abs(dist) - 0.5).cuda())
            # loss_mesh = loss_mesh_cls + loss_mesh_dist
            # dist = self.getDistances(cVertsGT, cVerts)*(-torch.log(cVertsProbM))*cVertsGTProbM
            # loss_mesh = torch.where(torch.abs(dist)<1., 0.5 * (dist ** 2), torch.abs(dist) - 0.5).cuda()
        else:
            cVerts, cVertsGT = self.findAllClosestVerts(u_gt, v_gt, i_gt, u_est, v_est, i_est)
            # vertexEst, vertexGT = self.getDistances(cVertsGT, cVerts) # J*3
            # print(vertexEst.shape, vertexGT.shape)
            # loss_U = F.smooth_l1_loss(u_est, u_gt, reduction="sum")
            # loss_V = F.smooth_l1_loss(v_est, v_gt, reduction="sum")
            # loss_mesh = F.smooth_l1_loss(vertexEst, vertexGT, reduction="sum")
            # index = (i_gt == i_est).nonzero().flatten().detach()
            # index_f = (i_gt != i_est).nonzero().flatten().detach()
            dist_mesh = torch.from_numpy(self.getDistances(cVertsGT, cVerts)).float().cuda() # J
            # dist_u = (u_gt-u_est).float()
            # dist_v = (v_gt-v_est).float()
            # dist_u = torch.where((i_gt==i_est).detach(), dist_u, dist_mesh)
            # dist_v = torch.where((i_gt==i_est).detach(), dist_v, dist_mesh)
            # loss_U = torch.where(torch.abs(dist_u)<1., 0.5 * (dist_u ** 2), torch.abs(dist_u) - 0.5).cuda()
            # loss_V = torch.where(torch.abs(dist_v)<1., 0.5 * (dist_v ** 2), torch.abs(dist_v) - 0.5).cuda()
            loss_mesh = torch.where(torch.abs(dist_mesh)<1., 0.5 * (dist_mesh ** 2), torch.abs(dist_mesh) - 0.5).cuda()
            loss_U = F.smooth_l1_loss(u_est, u_gt, reduction="sum")
            loss_V = F.smooth_l1_loss(v_est, v_gt, reduction="sum")
            # u_gt = u_gt[index]
            # v_gt = v_gt[index]
            # u_est = u_est[index]
            # v_est = v_est[index]
            # if len(index) > 0:
            #     loss_U = F.smooth_l1_loss(u_est, u_gt, reduction="sum")
            #     loss_V = F.smooth_l1_loss(v_est, v_gt, reduction="sum")
            # else:
            #     loss_U = densepose_predictor_outputs.u.sum() * 0.
            #     loss_V = densepose_predictor_outputs.v.sum() * 0.
        if self.mesh_uv_loss:
            return{
                "loss_densepose_U": loss_U * self.w_points,
                "loss_densepose_V": loss_V * self.w_points,
                "loss_densepose_mesh": torch.sum(loss_mesh) * self.w_mesh,
            }
        return {
            "loss_densepose_mesh": torch.sum(loss_mesh) * self.w_points,
        }

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        tensors_helper: SingleTensorsHelper,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Losses for fine / coarse segmentation: cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points for fine segmentation and dense mask annotations
        for coarse segmentation.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """
        fine_segm_gt = tensors_helper.fine_segm_labels_gt[interpolator.j_valid] # J
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm[tensors_helper.index_with_dp],
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[interpolator.j_valid, :]
        if self.eql:
            def point2mask(preds, labels, J):
                target = torch.zeros_like(preds, dtype=torch.float).cuda()
                target[torch.arange(J).cuda(), labels] = 1
                return target
            
            J = fine_segm_gt.shape[0]
            E = self.exclude_func(J)
            T = self.threshold_func(fine_segm_gt, J)
            y_t = point2mask(fine_segm_est, fine_segm_gt, J)
            M = torch.max(fine_segm_est, dim=1, keepdim=True)[0]
            # eql_w = 1 - E * T * (1 - y_t)
            eql_w = 1. - E * T * (1. - y_t)*torch.exp(-torch.abs(torch.true_divide(fine_segm_est, (M.repeat(1, fine_segm_est.shape[1])))))
            # sigmoid
            # cls_loss = F.binary_cross_entropy(fine_segm_est, fine_segm_gt.float(), weight=eql_w, reduction='none')
            # softmax
            # M = torch.max(fine_segm_est, dim=1, keepdim=True)[0]
            x = (fine_segm_est-M) - torch.log(torch.sum(eql_w*torch.exp(fine_segm_est-M), dim=1)).unsqueeze(1).repeat(1, fine_segm_est.shape[1])
            cls_loss = F.nll_loss(x, fine_segm_gt)

            # output_dir = "./output/densepose_rcnn_R_101_FPN_DL_MDN_s1x/"
            # if output_dir:
            #     file_path = os.path.join(output_dir, "loss.json")
            #     uv_bbox = {
            #         "fine_segm_gt": fine_segm_gt.cpu().numpy().tolist(),
            #         "eql_w": eql_w.cpu().numpy().tolist(),
            #         "cls_loss": cls_loss.detach().cpu().numpy().tolist(),
            #     }
            #     with open(file_path, "a") as f:
            #         json.dump(uv_bbox, f) 
            losses = {
                "loss_densepose_I": cls_loss * self.w_part
            }
        else:
            losses = {
                "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part
            }

        if not self.segm_trained_by_masks:
            # Resample everything to the estimated data size, no need to resample
            # S_est then:
            coarse_segm_est = densepose_predictor_outputs.coarse_segm[tensors_helper.index_with_dp]
            with torch.no_grad():
                coarse_segm_gt = resample_data(
                    tensors_helper.coarse_segm_gt.unsqueeze(1),
                    tensors_helper.bbox_xywh_gt,
                    tensors_helper.bbox_xywh_est,
                    self.heatmap_size,
                    self.heatmap_size,
                    mode="nearest",
                    padding_mode="zeros",
                ).squeeze(1)
            if self.n_segm_chan == 2:
                coarse_segm_gt = coarse_segm_gt > 0
            losses["loss_densepose_S"] = (
                F.cross_entropy(coarse_segm_est, coarse_segm_gt.long()) * self.w_segm
            )
        return losses
    def exclude_func(self, J):
        # sigmoid
        # bg_ind = 0
        # weight = (gt_classes != bg_ind).float()
        # weight = weight.unsqueeze(-1)
        # weight = weight.repeat(1, self.n_i_chan+1)
        # return weight.cuda()
        # softmax
        weight = torch.zeros((self.n_i_chan+1), dtype=torch.float).cuda()
        beta = torch.zeros_like(weight).cuda().uniform_()
        weight[beta < self.gamma] = 1
        weight = weight.view(1, self.n_i_chan+1).expand(J, self.n_i_chan+1)
        return weight

    def threshold_func(self, gt_classes, J): 
        weight = torch.zeros(self.n_i_chan+1).cuda()
        # freq = torch.zeros(25, dtype=float).cuda()
        # freq = torch.bincount(gt_classes, minlength=25)/torch.sum(torch.bincount(gt_classes, minlength=25))
        # weight[freq < self.eql_lambda] = 1 
        if self.threshold_func_type == "gompertz_decay":
            freq = torch.tensor([0,0.04157,0.1245,0.0554,0.0515,0.0310,0.0304,0.0164,0.0167,0.0531,0.0527,0.0143,0.0158,0.0385,0.0397,0.0340,0.0358,0.0419,0.0441,0.0207,0.0262,0.0486,0.0461,0.0623,0.0586], dtype=float).cuda()
            weight = 1-torch.exp(-8*torch.exp(-100*freq))
        if self.threshold_func_type == "exponential_filter":
            freq = torch.tensor([0,0.04157,0.1245,0.0554,0.0515,0.0310,0.0304,0.0164,0.0167,0.0531,0.0527,0.0143,0.0158,0.0385,0.0397,0.0340,0.0358,0.0419,0.0441,0.0207,0.0262,0.0486,0.0461,0.0623,0.0586], dtype=float).cuda()
            weight = 1-8*freq
        if self.freq is not None:
            for f in self.freq:
                weight[f] = 1
        weight = weight.unsqueeze(0)
        weight = weight.repeat(J, 1)
        if self.threshold_func_type == "relation_filter":
            relation_matrix = torch.tensor([0,2,1,4,3,6,5,9,10,7,8,13,14,11,12,17,18,15,16,21,22,19,20,24,23]).cuda()
            
            relation_gt = relation_matrix[gt_classes[torch.arange(J).cuda()]] 
            weight[torch.arange(J).cuda(), relation_gt] = 0
            # print(weight.cpu().numpy().tolist())
        return weight
    
    def findAllClosestVerts(self, U_gt, V_gt, I_gt, U_points, V_points, Index_points):

        ClosestVerts = torch.ones(Index_points.shape).cuda() * -1.
        # ProbVerts = torch.ones(Index_points.shape).cuda() * -1.
        if self.mesh_with_classify:
            ClosestVerts = torch.ones(Index_points.shape[0]).cuda() * -1.
            ProbVerts = torch.ones(Index_points.shape[0]).cuda() * -1.
        for i in range(24):
            #
            if (i + 1) in Index_points:
                UVs = torch.stack(( U_points[Index_points == (i + 1)], V_points[Index_points == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                # print(UVs.shape, Current_Part_UVs.shape)
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                # ProbVerts[Index_points == (i + 1), 0:(Current_Part_UVs.shape[1])] = D.transpose(1,0)
                
                if self.mesh_with_classify:
                    topk_value, top_index = torch.topk(D.squeeze().float(), 1, dim=0, largest=False, sorted=True)
                    # if len(top_index.shape) == 1:
                    #     top_index = top_index.unsqueeze(1)
                    #     topk_value = topk_value.unsqueeze(1)
                    ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[top_index]
                    ProbVerts[Index_points == (i + 1)] = torch.exp(-topk_value)
                    # ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[top_index[torch.arange(2)]].transpose(1,0)
                    # ProbVerts[Index_points == (i + 1)] = torch.exp(-topk_value.transpose(1,0))
                else:
                    ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[
                        torch.argmin(D.squeeze().float(), axis=0)
                    ]
        #
        ClosestVertsGT = torch.ones(Index_points.shape).cuda() * -1
        # ProbVertsGT = torch.ones(Index_points.shape).cuda() * -1
        for i in range(24):
            if (i + 1) in I_gt:
                UVs = torch.stack((U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                # ProbVertsGT[I_gt == (i + 1), 0:(Current_Part_UVs.shape[1])] = D.transpose(1,0)
                ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[torch.argmin(D.squeeze().float(), axis=0)]
                if self.mesh_with_classify:
                    topk_value, top_index = torch.topk(D.squeeze().float(), 1, dim=0, largest=False, sorted=True)
                    ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[top_index]
                    ProbVertsGT[I_gt == (i + 1)] = torch.exp(-topk_value)
        #
        if self.mesh_with_classify:
            return ClosestVerts, ClosestVertsGT, ProbVerts, ProbVertsGT
        return ClosestVerts, ClosestVertsGT

    def getDistances(self, cVertsGT, cVerts):

        ClosestVertsTransformed = self.PDIST_transform[cVerts.long() - 1]
        ClosestVertsGTTransformed = self.PDIST_transform[cVertsGT.long() - 1]
        #
        ClosestVertsTransformed[cVerts < 0] = 0
        ClosestVertsGTTransformed[cVertsGT < 0] = 0
        #
        cVertsGT = ClosestVertsGTTransformed
        cVerts = ClosestVertsTransformed
        #
        # vertexGT = self.Vertex.transpose(1,0)[cVertsGT] # 29408x3
        # vertexEst = self.Vertex.transpose(1,0)[cVerts] # 29408*3
        # return vertexEst, vertexGT
        n = 27554
        dists = []
        # index_cVertsGT = (cVertsGT > 0).nonzero().flatten().detach()
        # cVerts_filter = cVerts[index_cVertsGT]
        # cVertsGT = cVertsGT[index_cVertsGT]
        # # dists = torch.zeros(len(cVerts_filter), dtype=torch.float32)
        # # dists[cVerts_filter <= 0] = 3.
        # oulter = torch.arange(len((cVerts_filter <= 0).nonzero()))*0+3.
        # index_cVerts = (cVerts_filter > 0).nonzero().flatten().detach()
        # cVerts_filter = cVerts_filter[index_cVerts] - 1
        # cVertsGT_filter = cVertsGT[index_cVerts] - 1
        # dists = torch.zeros(len(cVerts_filter), dtype=torch.float32)

        # # cVerts_concat = np.stack((cVertsGT_filter[cVerts_filter != cVertsGT_filter], cVerts_filter[cVerts_filter != cVertsGT_filter]), dim=1)
        # cVerts_max = torch.max(cVertsGT_filter, cVerts_filter)
        # cVerts_min = torch.min(cVertsGT_filter, cVerts_filter)
        # dist_matrix = torch.true_divide(cVerts_max*(cVerts_max-1), 2) + cVerts_min
        # dists[cVerts_filter != cVertsGT_filter] = self.Pdist_matrix[dist_matrix[cVerts_filter != cVertsGT_filter].long()][0]
        # dists = torch.cat((dists, oulter), axis=0)
        for d in range(len(cVertsGT)):
            if cVertsGT[d] > 0:
                if cVerts[d] > 0:
                    i = cVertsGT[d] - 1
                    j = cVerts[d] - 1
                    if j == i:
                        dists.append(0)
                    elif j > i:
                        # ccc = i
                        # i = j
                        # j = ccc
                        k = torch.true_divide(j*(j-1), 2) + i
                        # i = n - i - 1
                        # j = n - j - 1
                        # k = torch.true_divide(n * (n - 1), 2) - torch.true_divide((n - i) * ((n - i) - 1), 2) + j - i - 1
                        # k = torch.true_divide((n * n - n), 2) - k - 1
                        if k > 379597680:
                            print(k, i ,j)
                            print(cVertsGT)
                            print(cVerts)
                        dists.append(self.Pdist_matrix[k.long()][0])
                    else:
                        k = torch.true_divide(i*(i-1), 2) + j
                        # i = n - i - 1
                        # j = n - j - 1
                        # k = torch.true_divide(n * (n - 1), 2) - torch.true_divide((n - i) * ((n - i) - 1), 2) + j - i - 1
                        # k = torch.true_divide((n * n - n), 2) - k - 1
                        if k > 379597680:
                            print(k, i ,j)
                            print(cVertsGT)
                            print(cVerts)
                        dists.append(self.Pdist_matrix[k.long()][0])
                else:
                    dists.append(3.)
        return np.atleast_1d(np.array(dists).squeeze())
