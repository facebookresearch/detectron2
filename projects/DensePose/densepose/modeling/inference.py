# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Tuple
import torch

from detectron2.structures import Instances

from ..data.structures import DensePoseOutput


def densepose_inference(
    densepose_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    densepose_confidences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    detections: List[Instances],
):
    """
    Infer dense pose estimate based on outputs from the DensePose head
    and detections. The estimate for each detection instance is stored in its
    "pred_densepose" attribute.

    Args:
        densepose_outputs (tuple(`torch.Tensor`)): iterable containing 4 elements:
            - s (:obj: `torch.Tensor`): coarse segmentation tensor of size (N, A, H, W),
            - i (:obj: `torch.Tensor`): fine segmentation tensor of size (N, C, H, W),
            - u (:obj: `torch.Tensor`): U coordinates for each class of size (N, C, H, W),
            - v (:obj: `torch.Tensor`): V coordinates for each class of size (N, C, H, W),
            where N is the total number of detections in a batch,
                  A is the number of coarse segmentations labels
                      (e.g. 15 for coarse body parts + background),
                  C is the number of fine segmentation labels
                      (e.g. 25 for fine body parts + background),
                  W is the resolution along the X axis
                  H is the resolution along the Y axis
        densepose_confidences (tuple(`torch.Tensor`)): iterable containing 4 elements:
            - sigma_1 (:obj: `torch.Tensor`): global confidences for UV coordinates
                of size (N, C, H, W)
            - sigma_2 (:obj: `torch.Tensor`): individual confidences for UV coordinates
                of size (N, C, H, W)
            - kappa_u (:obj: `torch.Tensor`): first component of confidence direction
                vector of size (N, C, H, W)
            - kappa_v (:obj: `torch.Tensor`): second component of confidence direction
                vector of size (N, C, H, W)
            - fine_segm_confidence (:obj: `torch.Tensor`): confidence for fine
                segmentation of size (N, 1, H, W)
            - coarse_segm_confidence (:obj: `torch.Tensor`): confidence for coarse
                segmentation of size (N, 1, H, W)
        detections (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Instances are modified by this method: "pred_densepose" attribute
            is added to each instance, the attribute contains the corresponding
            DensePoseOutput object.
    """
    # DensePose outputs: segmentation, body part indices, U, V
    s, index_uv, u, v = densepose_outputs
    (
        sigma_1,
        sigma_2,
        kappa_u,
        kappa_v,
        fine_segm_confidence,
        coarse_segm_confidence,
    ) = densepose_confidences
    k = 0
    for detection in detections:
        n_i = len(detection)
        s_i = s[k : k + n_i]
        index_uv_i = index_uv[k : k + n_i]
        u_i = u[k : k + n_i]
        v_i = v[k : k + n_i]
        _local_vars = locals()
        confidences = {
            name: _local_vars[name][k : k + n_i]
            for name in (
                "sigma_1",
                "sigma_2",
                "kappa_u",
                "kappa_v",
                "fine_segm_confidence",
                "coarse_segm_confidence",
            )
            if _local_vars.get(name) is not None
        }
        densepose_output_i = DensePoseOutput(s_i, index_uv_i, u_i, v_i, confidences)
        detection.pred_densepose = densepose_output_i
        k += n_i
