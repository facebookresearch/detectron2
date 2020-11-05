# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.structures import ImageList


@PROPOSAL_GENERATOR_REGISTRY.register()
class TridentRPN(RPN):
    """
    Trident RPN subnetwork.
    """

    def __init__(self, cfg, input_shape):
        super(TridentRPN, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, gt_instances=None):
        """
        See :class:`RPN.forward`.
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        # Duplicate images and gt_instances for all branches in TridentNet.
        all_images = ImageList(
            torch.cat([images.tensor] * num_branch), images.image_sizes * num_branch
        )
        all_gt_instances = gt_instances * num_branch if gt_instances is not None else None

        return super(TridentRPN, self).forward(all_images, features, all_gt_instances)
