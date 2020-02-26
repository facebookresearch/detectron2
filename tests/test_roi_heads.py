# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)


class ROIHeadsTest(unittest.TestCase):
    def test_roi_heads(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        backbone = build_backbone(cfg)
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}

        image_shape = (15, 15)
        gt_boxes0 = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        gt_instance0 = Instances(image_shape)
        gt_instance0.gt_boxes = Boxes(gt_boxes0)
        gt_instance0.gt_classes = torch.tensor([2, 1])
        gt_boxes1 = torch.tensor([[1, 5, 2, 8], [7, 3, 10, 5]], dtype=torch.float32)
        gt_instance1 = Instances(image_shape)
        gt_instance1.gt_boxes = Boxes(gt_boxes1)
        gt_instance1.gt_classes = torch.tensor([1, 2])
        gt_instances = [gt_instance0, gt_instance1]

        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
        roi_heads = build_roi_heads(cfg, backbone.output_shape())

        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(images, features, gt_instances)
            _, detector_losses = roi_heads(images, features, proposals, gt_instances)

        expected_losses = {
            "loss_cls": torch.tensor(4.4236516953),
            "loss_box_reg": torch.tensor(0.0091214813),
        }
        for name in expected_losses.keys():
            self.assertTrue(torch.allclose(detector_losses[name], expected_losses[name]))

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
        backbone = build_backbone(cfg)
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}

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

        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
        roi_heads = build_roi_heads(cfg, backbone.output_shape())

        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(images, features, gt_instances)
            _, detector_losses = roi_heads(images, features, proposals, gt_instances)

        expected_losses = {
            "loss_cls": torch.tensor(4.381443977355957),
            "loss_box_reg": torch.tensor(0.0011560433777049184),
        }
        for name in expected_losses.keys():
            err_msg = "detector_losses[{}] = {}, expected losses = {}".format(
                name, detector_losses[name], expected_losses[name]
            )
            self.assertTrue(torch.allclose(detector_losses[name], expected_losses[name]), err_msg)


if __name__ == "__main__":
    unittest.main()
