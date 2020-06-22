# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage

logger = logging.getLogger(__name__)


class RPNTest(unittest.TestCase):
    def test_rpn(self):
        torch.manual_seed(121)
        cfg = get_cfg()
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
        cfg.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1)
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
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
            "loss_rpn_cls": torch.tensor(0.0804563984),
            "loss_rpn_loc": torch.tensor(0.0990132466),
        }
        for name in expected_losses.keys():
            err_msg = "proposal_losses[{}] = {}, expected losses = {}".format(
                name, proposal_losses[name], expected_losses[name]
            )
            self.assertTrue(torch.allclose(proposal_losses[name], expected_losses[name]), err_msg)

        expected_proposal_boxes = [
            Boxes(torch.tensor([[0, 0, 10, 10], [7.3365392685, 0, 10, 10]])),
            Boxes(
                torch.tensor(
                    [
                        [0, 0, 30, 20],
                        [0, 0, 16.7862777710, 13.1362524033],
                        [0, 0, 30, 13.3173446655],
                        [0, 0, 10.8602609634, 20],
                        [7.7165775299, 0, 27.3875980377, 20],
                    ]
                )
            ),
        ]

        expected_objectness_logits = [
            torch.tensor([0.1225359365, -0.0133192837]),
            torch.tensor([0.1415634006, 0.0989848152, 0.0565387346, -0.0072308783, -0.0428492837]),
        ]

        for proposal, expected_proposal_box, im_size, expected_objectness_logit in zip(
            proposals, expected_proposal_boxes, image_sizes, expected_objectness_logits
        ):
            self.assertEqual(len(proposal), len(expected_proposal_box))
            self.assertEqual(proposal.image_size, im_size)
            self.assertTrue(
                torch.allclose(proposal.proposal_boxes.tensor, expected_proposal_box.tensor)
            )
            self.assertTrue(torch.allclose(proposal.objectness_logits, expected_objectness_logit))

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
            "loss_rpn_cls": torch.tensor(0.043263837695121765),
            "loss_rpn_loc": torch.tensor(0.14432406425476074),
        }
        for name in expected_losses.keys():
            err_msg = "proposal_losses[{}] = {}, expected losses = {}".format(
                name, proposal_losses[name], expected_losses[name]
            )
            self.assertTrue(torch.allclose(proposal_losses[name], expected_losses[name]), err_msg)

        expected_proposal_boxes = [
            RotatedBoxes(
                torch.tensor(
                    [
                        [0.60189795, 1.24095452, 61.98131943, 18.03621292, -4.07244873],
                        [15.64940453, 1.69624567, 59.59749603, 16.34339333, 2.62692475],
                        [-3.02982378, -2.69752932, 67.90952301, 59.62455750, 59.97010040],
                        [16.71863365, 1.98309708, 35.61507797, 32.81484985, 62.92267227],
                        [0.49432933, -7.92979717, 67.77606201, 62.93098450, -1.85656738],
                        [8.00880814, 1.36017394, 121.81007385, 32.74150467, 50.44297409],
                        [16.44299889, -4.82221127, 63.39775848, 61.22503662, 54.12270737],
                        [5.00000000, 5.00000000, 10.00000000, 10.00000000, -0.76943970],
                        [17.64130402, -0.98095351, 61.40377808, 16.28918839, 55.53118134],
                        [0.13016054, 4.60568953, 35.80157471, 32.30180359, 62.52872086],
                        [-4.26460743, 0.39604485, 124.30079651, 31.84611320, -1.58203125],
                        [7.52815342, -0.91636634, 62.39784622, 15.45565224, 60.79549789],
                    ]
                )
            ),
            RotatedBoxes(
                torch.tensor(
                    [
                        [0.07734215, 0.81635046, 65.33510590, 17.34688377, -1.51821899],
                        [-3.41833067, -3.11320257, 64.17595673, 60.55617905, 58.27033234],
                        [20.67383385, -6.16561556, 63.60531998, 62.52315903, 54.85546494],
                        [15.00000000, 10.00000000, 30.00000000, 20.00000000, -0.18218994],
                        [9.22646523, -6.84775209, 62.09895706, 65.46472931, -2.74307251],
                        [15.00000000, 4.93451595, 30.00000000, 9.86903191, -0.60272217],
                        [8.88342094, 2.65560246, 120.95362854, 32.45022202, 55.75970078],
                        [16.39088631, 2.33887148, 34.78761292, 35.61492920, 60.81977463],
                        [9.78298569, 10.00000000, 19.56597137, 20.00000000, -0.86660767],
                        [1.28576660, 5.49873352, 34.93610382, 33.22600174, 60.51599884],
                        [17.58912468, -1.63270092, 62.96052551, 16.45713997, 52.91245270],
                        [5.64749718, -1.90428460, 62.37649155, 16.19474792, 61.09543991],
                        [0.82255805, 2.34931135, 118.83985901, 32.83671188, 56.50753784],
                        [-5.33874989, 1.64404404, 125.28501892, 33.35424042, -2.80731201],
                    ]
                )
            ),
        ]

        expected_objectness_logits = [
            torch.tensor(
                [
                    0.10111768,
                    0.09112845,
                    0.08466332,
                    0.07589971,
                    0.06650183,
                    0.06350251,
                    0.04299347,
                    0.01864817,
                    0.00986163,
                    0.00078543,
                    -0.04573630,
                    -0.04799230,
                ]
            ),
            torch.tensor(
                [
                    0.11373727,
                    0.09377633,
                    0.05281663,
                    0.05143715,
                    0.04040275,
                    0.03250912,
                    0.01307789,
                    0.01177734,
                    0.00038105,
                    -0.00540255,
                    -0.01194804,
                    -0.01461012,
                    -0.03061717,
                    -0.03599222,
                ]
            ),
        ]

        torch.set_printoptions(precision=8, sci_mode=False)

        for proposal, expected_proposal_box, im_size, expected_objectness_logit in zip(
            proposals, expected_proposal_boxes, image_sizes, expected_objectness_logits
        ):
            self.assertEqual(len(proposal), len(expected_proposal_box))
            self.assertEqual(proposal.image_size, im_size)
            # It seems that there's some randomness in the result across different machines:
            # This test can be run on a local machine for 100 times with exactly the same result,
            # However, a different machine might produce slightly different results,
            # thus the atol here.
            err_msg = "computed proposal boxes = {}, expected {}".format(
                proposal.proposal_boxes.tensor, expected_proposal_box.tensor
            )
            self.assertTrue(
                torch.allclose(
                    proposal.proposal_boxes.tensor, expected_proposal_box.tensor, atol=1e-5
                ),
                err_msg,
            )

            err_msg = "computed objectness logits = {}, expected {}".format(
                proposal.objectness_logits, expected_objectness_logit
            )
            self.assertTrue(
                torch.allclose(proposal.objectness_logits, expected_objectness_logit, atol=1e-5),
                err_msg,
            )

    def test_rpn_proposals_inf(self):
        N, Hi, Wi, A = 3, 3, 3, 3
        proposals = [torch.rand(N, Hi * Wi * A, 4)]
        pred_logits = [torch.rand(N, Hi * Wi * A)]
        pred_logits[0][1][3:5].fill_(float("inf"))
        find_top_rpn_proposals(proposals, pred_logits, [(10, 10)], 0.5, 1000, 1000, 0, False)


if __name__ == "__main__":
    unittest.main()
