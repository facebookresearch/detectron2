# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads import StandardROIHeads, build_roi_heads
from detectron2.structures import BitMasks, Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)

"""
Make sure the losses of ROIHeads/RPN do not change, to avoid
breaking the forward logic by mistake.
This relies on assumption that pytorch's RNG is stable.
"""


class ROIHeadsTest(unittest.TestCase):
    def test_roi_heads(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        cfg.MODEL.MASK_ON = True
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        feature_shape = {"res4": ShapeSpec(channels=num_channels, stride=16)}

        image_shape = (15, 15)
        gt_boxes0 = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        gt_instance0 = Instances(image_shape)
        gt_instance0.gt_boxes = Boxes(gt_boxes0)
        gt_instance0.gt_classes = torch.tensor([2, 1])
        gt_instance0.gt_masks = BitMasks(torch.rand((2,) + image_shape) > 0.5)
        gt_boxes1 = torch.tensor([[1, 5, 2, 8], [7, 3, 10, 5]], dtype=torch.float32)
        gt_instance1 = Instances(image_shape)
        gt_instance1.gt_boxes = Boxes(gt_boxes1)
        gt_instance1.gt_classes = torch.tensor([1, 2])
        gt_instance1.gt_masks = BitMasks(torch.rand((2,) + image_shape) > 0.5)
        gt_instances = [gt_instance0, gt_instance1]

        proposal_generator = build_proposal_generator(cfg, feature_shape)
        roi_heads = StandardROIHeads(cfg, feature_shape)

        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(images, features, gt_instances)
            _, detector_losses = roi_heads(images, features, proposals, gt_instances)

        detector_losses.update(proposal_losses)
        expected_losses = {
            "loss_cls": 4.5253729820251465,
            "loss_box_reg": 0.009785720147192478,
            "loss_mask": 0.693184494972229,
            "loss_rpn_cls": 0.08186662942171097,
            "loss_rpn_loc": 0.1104838103055954,
        }
        succ = all(
            torch.allclose(detector_losses[name], torch.tensor(expected_losses.get(name, 0.0)))
            for name in detector_losses.keys()
        )
        self.assertTrue(
            succ,
            "Losses has changed! New losses: {}".format(
                {k: v.item() for k, v in detector_losses.items()}
            ),
        )

    def test_rroi_heads(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
        cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        feature_shape = {"res4": ShapeSpec(channels=num_channels, stride=16)}

        image_shape = (15, 15)
        gt_boxes0 = torch.tensor([[2, 2, 2, 2, 30], [4, 4, 4, 4, 0]], dtype=torch.float32)
        gt_instance0 = Instances(image_shape)
        gt_instance0.gt_boxes = RotatedBoxes(gt_boxes0)
        gt_instance0.gt_classes = torch.tensor([2, 1])
        gt_boxes1 = torch.tensor([[1.5, 5.5, 1, 3, 0], [8.5, 4, 3, 2, -50]], dtype=torch.float32)
        gt_instance1 = Instances(image_shape)
        gt_instance1.gt_boxes = RotatedBoxes(gt_boxes1)
        gt_instance1.gt_classes = torch.tensor([1, 2])
        gt_instances = [gt_instance0, gt_instance1]

        proposal_generator = build_proposal_generator(cfg, feature_shape)
        roi_heads = build_roi_heads(cfg, feature_shape)

        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(images, features, gt_instances)
            _, detector_losses = roi_heads(images, features, proposals, gt_instances)

        detector_losses.update(proposal_losses)
        expected_losses = {
            "loss_cls": 4.365657806396484,
            "loss_box_reg": 0.0015851043863222003,
            "loss_rpn_cls": 0.2427729219198227,
            "loss_rpn_loc": 0.3646621108055115,
        }
        succ = all(
            torch.allclose(detector_losses[name], torch.tensor(expected_losses.get(name, 0.0)))
            for name in detector_losses.keys()
        )
        self.assertTrue(
            succ,
            "Losses has changed! New losses: {}".format(
                {k: v.item() for k, v in detector_losses.items()}
            ),
        )


if __name__ == "__main__":
    unittest.main()
