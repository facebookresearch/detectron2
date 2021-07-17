# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Any, List

from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead, mask_rcnn_inference
from detectron2.projects.point_rend import ImplicitPointRendMaskHead
from detectron2.projects.point_rend.point_features import point_sample
from detectron2.projects.point_rend.point_head import roi_mask_point_loss
from detectron2.structures import Instances

from .point_utils import get_point_coords_from_point_annotation

__all__ = [
    "ImplicitPointRendPointSupHead",
    "MaskRCNNConvUpsamplePointSupHead",
]


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsamplePointSupHead(MaskRCNNConvUpsampleHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.

    The difference with `MaskRCNNConvUpsampleHead` is that this head is trained
    with point supervision. Please use the `MaskRCNNConvUpsampleHead` if you want
    to train the model with mask supervision.
    """

    def forward(self, x, instances: List[Instances]) -> Any:
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            N, C, H, W = x.shape
            assert H == W

            proposal_boxes = [x.proposal_boxes for x in instances]
            assert N == np.sum(len(x) for x in proposal_boxes)

            if N == 0:
                return {"loss_mask": x.sum() * 0}

            # Training with point supervision
            point_coords, point_labels = get_point_coords_from_point_annotation(instances)

            mask_logits = point_sample(
                x,
                point_coords,
                align_corners=False,
            )

            return {"loss_mask": roi_mask_point_loss(mask_logits, instances, point_labels)}
        else:
            mask_rcnn_inference(x, instances)
            return instances


@ROI_MASK_HEAD_REGISTRY.register()
class ImplicitPointRendPointSupHead(ImplicitPointRendMaskHead):
    def _uniform_sample_train_points(self, instances):
        assert self.training
        # Please keep in mind that "gt_masks" is not used in this mask head.
        point_coords, point_labels = get_point_coords_from_point_annotation(instances)

        return point_coords, point_labels
