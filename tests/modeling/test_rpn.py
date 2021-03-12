# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import sys
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.export import scripting_with_instances
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import RPN, build_proposal_generator
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)


class RPNTest(unittest.TestCase):
    def test_rpn(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        backbone = build_backbone(cfg)
        proposal_generator = RPN(cfg, backbone.output_shape())
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        image_shape = (15, 15)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        gt_boxes = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        gt_instances = Instances(image_shape)
        gt_instances.gt_boxes = Boxes(gt_boxes)
        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(
                images, features, [gt_instances[0], gt_instances[1]]
            )

        expected_losses = {
            "loss_rpn_cls": torch.tensor(0.08011703193),
            "loss_rpn_loc": torch.tensor(0.101470276),
        }
        for name in expected_losses.keys():
            err_msg = "proposal_losses[{}] = {}, expected losses = {}".format(
                name, proposal_losses[name], expected_losses[name]
            )
            self.assertTrue(torch.allclose(proposal_losses[name], expected_losses[name]), err_msg)

        self.assertEqual(len(proposals), len(image_sizes))
        for proposal, im_size in zip(proposals, image_sizes):
            self.assertEqual(proposal.image_size, im_size)

        expected_proposal_box = torch.tensor([[0, 0, 10, 10], [7.2702, 0, 10, 10]])
        expected_objectness_logit = torch.tensor([0.1596, -0.0007])
        self.assertTrue(
            torch.allclose(proposals[0].proposal_boxes.tensor, expected_proposal_box, atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(proposals[0].objectness_logits, expected_objectness_logit, atol=1e-4)
        )

    # https://github.com/pytorch/pytorch/issues/46964
    @unittest.skipIf(
        TORCH_VERSION < (1, 7) or sys.version_info.minor <= 6, "Insufficient pytorch version"
    )
    def test_rpn_scriptability(self):
        cfg = get_cfg()
        proposal_generator = RPN(cfg, {"res4": ShapeSpec(channels=1024, stride=16)}).eval()
        num_images = 2
        images_tensor = torch.rand(num_images, 30, 40)
        image_sizes = [(32, 32), (30, 40)]
        images = ImageList(images_tensor, image_sizes)
        features = {"res4": torch.rand(num_images, 1024, 1, 2)}

        fields = {"proposal_boxes": Boxes, "objectness_logits": torch.Tensor}
        proposal_generator_ts = scripting_with_instances(proposal_generator, fields)

        proposals, _ = proposal_generator(images, features)
        proposals_ts, _ = proposal_generator_ts(images, features)

        for proposal, proposal_ts in zip(proposals, proposals_ts):
            self.assertEqual(proposal.image_size, proposal_ts.image_size)
            self.assertTrue(
                torch.equal(proposal.proposal_boxes.tensor, proposal_ts.proposal_boxes.tensor)
            )
            self.assertTrue(torch.equal(proposal.objectness_logits, proposal_ts.objectness_logits))

    def test_rrpn(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 1]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0, 60]]
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)
        image_shape = (15, 15)
        num_channels = 1024
        features = {"res4": torch.rand(num_images, num_channels, 1, 2)}
        gt_boxes = torch.tensor([[2, 2, 2, 2, 0], [4, 4, 4, 4, 0]], dtype=torch.float32)
        gt_instances = Instances(image_shape)
        gt_instances.gt_boxes = RotatedBoxes(gt_boxes)
        with EventStorage():  # capture events in a new storage to discard them
            proposals, proposal_losses = proposal_generator(
                images, features, [gt_instances[0], gt_instances[1]]
            )

        expected_losses = {
            "loss_rpn_cls": torch.tensor(0.04291602224),
            "loss_rpn_loc": torch.tensor(0.145077362),
        }
        for name in expected_losses.keys():
            err_msg = "proposal_losses[{}] = {}, expected losses = {}".format(
                name, proposal_losses[name], expected_losses[name]
            )
            self.assertTrue(torch.allclose(proposal_losses[name], expected_losses[name]), err_msg)

        expected_proposal_box = torch.tensor(
            [
                [-1.77999556, 0.78155339, 68.04367828, 14.78156471, 60.59333801],
                [13.82740974, -1.50282836, 34.67269897, 29.19676590, -3.81942749],
                [8.10392570, -0.99071521, 145.39100647, 32.13126373, 3.67242432],
                [5.00000000, 4.57370186, 10.00000000, 9.14740372, 0.89196777],
            ]
        )

        expected_objectness_logit = torch.tensor([0.10924313, 0.09881870, 0.07649877, 0.05858029])

        torch.set_printoptions(precision=8, sci_mode=False)

        self.assertEqual(len(proposals), len(image_sizes))

        proposal = proposals[0]
        # It seems that there's some randomness in the result across different machines:
        # This test can be run on a local machine for 100 times with exactly the same result,
        # However, a different machine might produce slightly different results,
        # thus the atol here.
        err_msg = "computed proposal boxes = {}, expected {}".format(
            proposal.proposal_boxes.tensor, expected_proposal_box
        )
        self.assertTrue(
            torch.allclose(proposal.proposal_boxes.tensor[:4], expected_proposal_box, atol=1e-5),
            err_msg,
        )

        err_msg = "computed objectness logits = {}, expected {}".format(
            proposal.objectness_logits, expected_objectness_logit
        )
        self.assertTrue(
            torch.allclose(proposal.objectness_logits[:4], expected_objectness_logit, atol=1e-5),
            err_msg,
        )

    def test_find_rpn_proposals_inf(self):
        N, Hi, Wi, A = 3, 3, 3, 3
        proposals = [torch.rand(N, Hi * Wi * A, 4)]
        pred_logits = [torch.rand(N, Hi * Wi * A)]
        pred_logits[0][1][3:5].fill_(float("inf"))
        find_top_rpn_proposals(proposals, pred_logits, [(10, 10)], 0.5, 1000, 1000, 0, False)

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_find_rpn_proposals_tracing(self):
        N, Hi, Wi, A = 3, 50, 50, 9
        proposal = torch.rand(N, Hi * Wi * A, 4)
        pred_logit = torch.rand(N, Hi * Wi * A)

        def func(proposal, logit, image_size):
            r = find_top_rpn_proposals(
                [proposal], [logit], [image_size], 0.7, 1000, 1000, 0, False
            )[0]
            size = r.image_size
            if not isinstance(size, torch.Tensor):
                size = torch.tensor(size)
            return (size, r.proposal_boxes.tensor, r.objectness_logits)

        other_inputs = []
        # test that it generalizes to other shapes
        for Hi, Wi, shp in [(30, 30, 60), (10, 10, 800)]:
            other_inputs.append(
                (
                    torch.rand(N, Hi * Wi * A, 4),
                    torch.rand(N, Hi * Wi * A),
                    torch.tensor([shp, shp]),
                )
            )
        torch.jit.trace(
            func, (proposal, pred_logit, torch.tensor([100, 100])), check_inputs=other_inputs
        )


if __name__ == "__main__":
    unittest.main()
