# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
from copy import deepcopy
import torch

from detectron2.config import get_cfg
from detectron2.export.torchscript import patch_instances
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    KRCNNConvDeconvUpsampleHead,
    MaskRCNNConvUpsampleHead,
    StandardROIHeads,
    build_roi_heads,
)
from detectron2.structures import BitMasks, Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.env import TORCH_VERSION
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

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_box_head_scriptability(self):
        input_shape = ShapeSpec(channels=1024, height=14, width=14)
        box_features = torch.randn(4, 1024, 14, 14)

        box_head = FastRCNNConvFCHead(
            input_shape, conv_dims=[512, 512], fc_dims=[1024, 1024]
        ).eval()
        script_box_head = torch.jit.script(box_head)

        origin_output = box_head(box_features)
        script_output = script_box_head(box_features)
        self.assertTrue(torch.equal(origin_output, script_output))

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_mask_head_scriptability(self):
        input_shape = ShapeSpec(channels=1024)
        mask_features = torch.randn(4, 1024, 14, 14)

        image_shapes = [(10, 10), (15, 15)]
        pred_instance0 = Instances(image_shapes[0])
        pred_classes0 = torch.tensor([1, 2, 3], dtype=torch.int64)
        pred_instance0.pred_classes = pred_classes0
        pred_instance1 = Instances(image_shapes[1])
        pred_classes1 = torch.tensor([4], dtype=torch.int64)
        pred_instance1.pred_classes = pred_classes1

        mask_head = MaskRCNNConvUpsampleHead(
            input_shape, num_classes=80, conv_dims=[256, 256]
        ).eval()
        # pred_instance will be in-place changed during the inference
        # process of `MaskRCNNConvUpsampleHead`
        origin_outputs = mask_head(mask_features, deepcopy([pred_instance0, pred_instance1]))

        fields = {"pred_masks": "Tensor", "pred_classes": "Tensor"}
        with patch_instances(fields) as NewInstances:
            sciript_mask_head = torch.jit.script(mask_head)
            pred_instance0 = NewInstances.from_instances(pred_instance0)
            pred_instance1 = NewInstances.from_instances(pred_instance1)
            script_outputs = sciript_mask_head(mask_features, [pred_instance0, pred_instance1])

        for origin_ins, script_ins in zip(origin_outputs, script_outputs):
            self.assertEqual(origin_ins.image_size, script_ins.image_size)
            self.assertTrue(torch.equal(origin_ins.pred_classes, script_ins.pred_classes))
            self.assertTrue(torch.equal(origin_ins.pred_masks, script_ins.pred_masks))

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_keypoint_head_scriptability(self):
        input_shape = ShapeSpec(channels=1024, height=14, width=14)
        keypoint_features = torch.randn(4, 1024, 14, 14)

        image_shapes = [(10, 10), (15, 15)]
        pred_boxes0 = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6], [1, 5, 2, 8]], dtype=torch.float32)
        pred_instance0 = Instances(image_shapes[0])
        pred_instance0.pred_boxes = Boxes(pred_boxes0)
        pred_boxes1 = torch.tensor([[7, 3, 10, 5]], dtype=torch.float32)
        pred_instance1 = Instances(image_shapes[1])
        pred_instance1.pred_boxes = Boxes(pred_boxes1)

        keypoint_head = KRCNNConvDeconvUpsampleHead(
            input_shape, num_keypoints=17, conv_dims=[512, 512]
        ).eval()
        origin_outputs = keypoint_head(
            keypoint_features, deepcopy([pred_instance0, pred_instance1])
        )

        fields = {
            "pred_boxes": "Boxes",
            "pred_keypoints": "Tensor",
            "pred_keypoint_heatmaps": "Tensor",
        }
        with patch_instances(fields) as NewInstances:
            sciript_keypoint_head = torch.jit.script(keypoint_head)
            pred_instance0 = NewInstances.from_instances(pred_instance0)
            pred_instance1 = NewInstances.from_instances(pred_instance1)
            script_outputs = sciript_keypoint_head(
                keypoint_features, [pred_instance0, pred_instance1]
            )

        for origin_ins, script_ins in zip(origin_outputs, script_outputs):
            self.assertEqual(origin_ins.image_size, script_ins.image_size)
            self.assertTrue(torch.equal(origin_ins.pred_keypoints, script_ins.pred_keypoints))
            self.assertTrue(
                torch.equal(origin_ins.pred_keypoint_heatmaps, script_ins.pred_keypoint_heatmaps)
            )


if __name__ == "__main__":
    unittest.main()
