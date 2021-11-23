# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import unittest
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform, Box2BoxTransformRotated
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.rotated_fast_rcnn import RotatedFastRCNNOutputLayers
from detectron2.structures import Boxes, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)


class FastRCNNTest(unittest.TestCase):
    def test_fast_rcnn(self):
        torch.manual_seed(132)

        box_head_output_size = 8

        box_predictor = FastRCNNOutputLayers(
            ShapeSpec(channels=box_head_output_size),
            box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
            num_classes=5,
        )
        feature_pooled = torch.rand(2, box_head_output_size)
        predictions = box_predictor(feature_pooled)

        proposal_boxes = torch.tensor([[0.8, 1.1, 3.2, 2.8], [2.3, 2.5, 7, 8]], dtype=torch.float32)
        gt_boxes = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        proposal = Instances((10, 10))
        proposal.proposal_boxes = Boxes(proposal_boxes)
        proposal.gt_boxes = Boxes(gt_boxes)
        proposal.gt_classes = torch.tensor([1, 2])

        with EventStorage():  # capture events in a new storage to discard them
            losses = box_predictor.losses(predictions, [proposal])

        expected_losses = {
            "loss_cls": torch.tensor(1.7951188087),
            "loss_box_reg": torch.tensor(4.0357131958),
        }
        for name in expected_losses.keys():
            assert torch.allclose(losses[name], expected_losses[name])

    def test_fast_rcnn_empty_batch(self, device="cpu"):
        box_predictor = FastRCNNOutputLayers(
            ShapeSpec(channels=10),
            box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
            num_classes=8,
        ).to(device=device)

        logits = torch.randn(0, 100, requires_grad=True, device=device)
        deltas = torch.randn(0, 4, requires_grad=True, device=device)
        losses = box_predictor.losses([logits, deltas], [])
        for value in losses.values():
            self.assertTrue(torch.allclose(value, torch.zeros_like(value)))
        sum(losses.values()).backward()
        self.assertTrue(logits.grad is not None)
        self.assertTrue(deltas.grad is not None)

        predictions, _ = box_predictor.inference([logits, deltas], [])
        self.assertEqual(len(predictions), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fast_rcnn_empty_batch_cuda(self):
        self.test_fast_rcnn_empty_batch(device=torch.device("cuda"))

    def test_fast_rcnn_rotated(self):
        torch.manual_seed(132)
        box_head_output_size = 8

        box_predictor = RotatedFastRCNNOutputLayers(
            ShapeSpec(channels=box_head_output_size),
            box2box_transform=Box2BoxTransformRotated(weights=(10, 10, 5, 5, 1)),
            num_classes=5,
        )
        feature_pooled = torch.rand(2, box_head_output_size)
        predictions = box_predictor(feature_pooled)
        proposal_boxes = torch.tensor(
            [[2, 1.95, 2.4, 1.7, 0], [4.65, 5.25, 4.7, 5.5, 0]], dtype=torch.float32
        )
        gt_boxes = torch.tensor([[2, 2, 2, 2, 0], [4, 4, 4, 4, 0]], dtype=torch.float32)
        proposal = Instances((10, 10))
        proposal.proposal_boxes = RotatedBoxes(proposal_boxes)
        proposal.gt_boxes = RotatedBoxes(gt_boxes)
        proposal.gt_classes = torch.tensor([1, 2])

        with EventStorage():  # capture events in a new storage to discard them
            losses = box_predictor.losses(predictions, [proposal])

        # Note: the expected losses are slightly different even if
        # the boxes are essentially the same as in the FastRCNNOutput test, because
        # bbox_pred in FastRCNNOutputLayers have different Linear layers/initialization
        # between the two cases.
        expected_losses = {
            "loss_cls": torch.tensor(1.7920907736),
            "loss_box_reg": torch.tensor(4.0410838127),
        }
        for name in expected_losses.keys():
            assert torch.allclose(losses[name], expected_losses[name])

    def test_predict_boxes_tracing(self):
        class Model(torch.nn.Module):
            def __init__(self, output_layer):
                super(Model, self).__init__()
                self._output_layer = output_layer

            def forward(self, proposal_deltas, proposal_boxes):
                instances = Instances((10, 10))
                instances.proposal_boxes = Boxes(proposal_boxes)
                return self._output_layer.predict_boxes((None, proposal_deltas), [instances])

        box_head_output_size = 8

        box_predictor = FastRCNNOutputLayers(
            ShapeSpec(channels=box_head_output_size),
            box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
            num_classes=5,
        )

        model = Model(box_predictor)

        from detectron2.export.torchscript_patch import patch_builtin_len

        with torch.no_grad(), patch_builtin_len():
            func = torch.jit.trace(model, (torch.randn(10, 20), torch.randn(10, 4)))

            o = func(torch.randn(10, 20), torch.randn(10, 4))
            self.assertEqual(o[0].shape, (10, 20))
            o = func(torch.randn(5, 20), torch.randn(5, 4))
            self.assertEqual(o[0].shape, (5, 20))
            o = func(torch.randn(20, 20), torch.randn(20, 4))
            self.assertEqual(o[0].shape, (20, 20))

    def test_predict_probs_tracing(self):
        class Model(torch.nn.Module):
            def __init__(self, output_layer):
                super(Model, self).__init__()
                self._output_layer = output_layer

            def forward(self, scores, proposal_boxes):
                instances = Instances((10, 10))
                instances.proposal_boxes = Boxes(proposal_boxes)
                return self._output_layer.predict_probs((scores, None), [instances])

        box_head_output_size = 8

        box_predictor = FastRCNNOutputLayers(
            ShapeSpec(channels=box_head_output_size),
            box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
            num_classes=5,
        )

        model = Model(box_predictor)

        from detectron2.export.torchscript_patch import patch_builtin_len

        with torch.no_grad(), patch_builtin_len():
            func = torch.jit.trace(model, (torch.randn(10, 6), torch.rand(10, 4)))
            o = func(torch.randn(10, 6), torch.randn(10, 4))
            self.assertEqual(o[0].shape, (10, 6))
            o = func(torch.randn(5, 6), torch.randn(5, 4))
            self.assertEqual(o[0].shape, (5, 6))
            o = func(torch.randn(20, 6), torch.randn(20, 4))
            self.assertEqual(o[0].shape, (20, 6))


if __name__ == "__main__":
    unittest.main()
