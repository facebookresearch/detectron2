# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import BilinearInterpolationHelper, LossDict, SingleTensorsHelper, resample_data


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
class DensePoseChartLoss:
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
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS

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
        return {
            "loss_densepose_U": densepose_predictor_outputs.u.sum() * 0,
            "loss_densepose_V": densepose_predictor_outputs.v.sum() * 0,
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
        u_est = interpolator.extract_at_points(
            densepose_predictor_outputs.u[tensors_helper.index_with_dp]
        )[j_valid_fg]

        v_gt = tensors_helper.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points(
            densepose_predictor_outputs.v[tensors_helper.index_with_dp]
        )[j_valid_fg]
        return {
            "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
            "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
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
        fine_segm_gt = tensors_helper.fine_segm_labels_gt[interpolator.j_valid]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm[tensors_helper.index_with_dp],
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[interpolator.j_valid, :]
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
