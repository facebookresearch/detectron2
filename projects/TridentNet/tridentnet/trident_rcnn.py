# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import batched_nms
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.structures import Instances


def merge_branch_instances(instances, num_branch, nms_thrsh, topk_per_image):
    """
    Merge detection results from different branches of TridentNet.
    Return detection results by applying non-maximum suppression (NMS) on bounding boxes
    and keep the unsuppressed boxes and other instances (e.g mask) if any.

    Args:
        instances (list[Instances]): A list of N * num_branch instances that store detection
            results. Contain N images and each image has num_branch instances.
        num_branch (int): Number of branches used for merging detection results for each image.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        results: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections after merging results from multiple
            branches.
    """
    if num_branch == 1:
        return instances

    batch_size = len(instances) // num_branch
    results = []
    for i in range(batch_size):
        instance = Instances.cat([instances[i + batch_size * j] for j in range(num_branch)])

        # Apply per-class NMS
        keep = batched_nms(
            instance.pred_boxes.tensor, instance.scores, instance.pred_classes, nms_thrsh
        )
        keep = keep[:topk_per_image]
        result = instance[keep]

        results.append(result)

    return results


@ROI_HEADS_REGISTRY.register()
class TridentRes5ROIHeads(Res5ROIHeads):
    """
    The TridentNet ROIHeads in a typical "C4" R-CNN model.
    See :class:`Res5ROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class TridentStandardROIHeads(StandardROIHeads):
    """
    The `StandardROIHeads` for TridentNet.
    See :class:`StandardROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super(TridentStandardROIHeads, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        # Use 1 branch if using trident_fast during inference.
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        # Duplicate targets for all branches in TridentNet.
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}
