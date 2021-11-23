# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import unittest
from copy import deepcopy
import torch
from torch import nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.torchscript_patch import (
    freeze_training_mode,
    patch_builtin_len,
    patch_instances,
)
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    KRCNNConvDeconvUpsampleHead,
    MaskRCNNConvUpsampleHead,
    StandardROIHeads,
    build_roi_heads,
)
from detectron2.projects import point_rend
from detectron2.structures import BitMasks, Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage
from detectron2.utils.testing import assert_instances_allclose, random_boxes

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

        fields = {"pred_masks": torch.Tensor, "pred_classes": torch.Tensor}
        with freeze_training_mode(mask_head), patch_instances(fields) as NewInstances:
            sciript_mask_head = torch.jit.script(mask_head)
            pred_instance0 = NewInstances.from_instances(pred_instance0)
            pred_instance1 = NewInstances.from_instances(pred_instance1)
            script_outputs = sciript_mask_head(mask_features, [pred_instance0, pred_instance1])

        for origin_ins, script_ins in zip(origin_outputs, script_outputs):
            assert_instances_allclose(origin_ins, script_ins, rtol=0)

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
            "pred_boxes": Boxes,
            "pred_keypoints": torch.Tensor,
            "pred_keypoint_heatmaps": torch.Tensor,
        }
        with freeze_training_mode(keypoint_head), patch_instances(fields) as NewInstances:
            sciript_keypoint_head = torch.jit.script(keypoint_head)
            pred_instance0 = NewInstances.from_instances(pred_instance0)
            pred_instance1 = NewInstances.from_instances(pred_instance1)
            script_outputs = sciript_keypoint_head(
                keypoint_features, [pred_instance0, pred_instance1]
            )

        for origin_ins, script_ins in zip(origin_outputs, script_outputs):
            assert_instances_allclose(origin_ins, script_ins, rtol=0)

    def test_StandardROIHeads_scriptability(self):
        cfg = get_cfg()
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        feature_shape = {"res4": ShapeSpec(channels=num_channels, stride=16)}

        roi_heads = StandardROIHeads(cfg, feature_shape).eval()

        proposal0 = Instances(image_sizes[0])
        proposal_boxes0 = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        proposal0.proposal_boxes = Boxes(proposal_boxes0)
        proposal0.objectness_logits = torch.tensor([0.5, 0.7], dtype=torch.float32)

        proposal1 = Instances(image_sizes[1])
        proposal_boxes1 = torch.tensor([[1, 5, 2, 8], [7, 3, 10, 5]], dtype=torch.float32)
        proposal1.proposal_boxes = Boxes(proposal_boxes1)
        proposal1.objectness_logits = torch.tensor([0.1, 0.9], dtype=torch.float32)
        proposals = [proposal0, proposal1]

        pred_instances, _ = roi_heads(images, features, proposals)
        fields = {
            "objectness_logits": torch.Tensor,
            "proposal_boxes": Boxes,
            "pred_classes": torch.Tensor,
            "scores": torch.Tensor,
            "pred_masks": torch.Tensor,
            "pred_boxes": Boxes,
            "pred_keypoints": torch.Tensor,
            "pred_keypoint_heatmaps": torch.Tensor,
        }
        with freeze_training_mode(roi_heads), patch_instances(fields) as new_instances:
            proposal0 = new_instances.from_instances(proposal0)
            proposal1 = new_instances.from_instances(proposal1)
            proposals = [proposal0, proposal1]
            scripted_rot_heads = torch.jit.script(roi_heads)
            scripted_pred_instances, _ = scripted_rot_heads(images, features, proposals)

        for instance, scripted_instance in zip(pred_instances, scripted_pred_instances):
            assert_instances_allclose(instance, scripted_instance, rtol=0)

    def test_PointRend_mask_head_tracing(self):
        cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        point_rend.add_pointrend_config(cfg)
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3"]
        cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
        cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""
        cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
        chan = 256
        head = point_rend.PointRendMaskHead(
            cfg,
            {
                "p2": ShapeSpec(channels=chan, stride=4),
                "p3": ShapeSpec(channels=chan, stride=8),
            },
        )

        def gen_inputs(h, w, N):
            p2 = torch.rand(1, chan, h, w)
            p3 = torch.rand(1, chan, h // 2, w // 2)
            boxes = random_boxes(N, max_coord=h)
            return p2, p3, boxes

        class Wrap(nn.ModuleDict):
            def forward(self, p2, p3, boxes):
                features = {
                    "p2": p2,
                    "p3": p3,
                }
                inst = Instances((p2.shape[2] * 4, p2.shape[3] * 4))
                inst.pred_boxes = Boxes(boxes)
                inst.pred_classes = torch.zeros(inst.__len__(), dtype=torch.long)
                out = self.head(features, [inst])[0]
                return out.pred_masks

        model = Wrap({"head": head})
        model.eval()
        with torch.no_grad(), patch_builtin_len():
            traced = torch.jit.trace(model, gen_inputs(302, 208, 20))
            inputs = gen_inputs(100, 120, 30)
            out_eager = model(*inputs)
            out_trace = traced(*inputs)
            self.assertTrue(torch.allclose(out_eager, out_trace))


if __name__ == "__main__":
    unittest.main()
