# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, RotatedAnchorGenerator
from detectron2.utils.env import TORCH_VERSION

logger = logging.getLogger(__name__)


class TestAnchorGenerator(unittest.TestCase):
    def test_default_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 1, 4]]

        anchor_generator = DefaultAnchorGenerator(cfg, [ShapeSpec(stride=4)])

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        anchors = anchor_generator([features["stage3"]])
        expected_anchor_tensor = torch.tensor(
            [
                [-32.0, -8.0, 32.0, 8.0],
                [-16.0, -16.0, 16.0, 16.0],
                [-8.0, -32.0, 8.0, 32.0],
                [-64.0, -16.0, 64.0, 16.0],
                [-32.0, -32.0, 32.0, 32.0],
                [-16.0, -64.0, 16.0, 64.0],
                [-28.0, -8.0, 36.0, 8.0],  # -28.0 == -32.0 + STRIDE (4)
                [-12.0, -16.0, 20.0, 16.0],
                [-4.0, -32.0, 12.0, 32.0],
                [-60.0, -16.0, 68.0, 16.0],
                [-28.0, -32.0, 36.0, 32.0],
                [-12.0, -64.0, 20.0, 64.0],
            ]
        )

        assert torch.allclose(anchors[0].tensor, expected_anchor_tensor)

    def test_default_anchor_generator_centered(self):
        # test explicit args
        anchor_generator = DefaultAnchorGenerator(
            sizes=[32, 64], aspect_ratios=[0.25, 1, 4], strides=[4]
        )

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        expected_anchor_tensor = torch.tensor(
            [
                [-30.0, -6.0, 34.0, 10.0],
                [-14.0, -14.0, 18.0, 18.0],
                [-6.0, -30.0, 10.0, 34.0],
                [-62.0, -14.0, 66.0, 18.0],
                [-30.0, -30.0, 34.0, 34.0],
                [-14.0, -62.0, 18.0, 66.0],
                [-26.0, -6.0, 38.0, 10.0],
                [-10.0, -14.0, 22.0, 18.0],
                [-2.0, -30.0, 14.0, 34.0],
                [-58.0, -14.0, 70.0, 18.0],
                [-26.0, -30.0, 38.0, 34.0],
                [-10.0, -62.0, 22.0, 66.0],
            ]
        )

        anchors = anchor_generator([features["stage3"]])
        assert torch.allclose(anchors[0].tensor, expected_anchor_tensor)

        if TORCH_VERSION >= (1, 6):
            anchors = torch.jit.script(anchor_generator)([features["stage3"]])
            assert torch.allclose(anchors[0].tensor, expected_anchor_tensor)

    def test_rrpn_anchor_generator(self):
        cfg = get_cfg()
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 1, 4]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [0, 45]  # test single list[float]
        anchor_generator = RotatedAnchorGenerator(cfg, [ShapeSpec(stride=4)])

        # only the last two dimensions of features matter here
        num_images = 2
        features = {"stage3": torch.rand(num_images, 96, 1, 2)}
        anchors = anchor_generator([features["stage3"]])
        expected_anchor_tensor = torch.tensor(
            [
                [0.0, 0.0, 64.0, 16.0, 0.0],
                [0.0, 0.0, 64.0, 16.0, 45.0],
                [0.0, 0.0, 32.0, 32.0, 0.0],
                [0.0, 0.0, 32.0, 32.0, 45.0],
                [0.0, 0.0, 16.0, 64.0, 0.0],
                [0.0, 0.0, 16.0, 64.0, 45.0],
                [0.0, 0.0, 128.0, 32.0, 0.0],
                [0.0, 0.0, 128.0, 32.0, 45.0],
                [0.0, 0.0, 64.0, 64.0, 0.0],
                [0.0, 0.0, 64.0, 64.0, 45.0],
                [0.0, 0.0, 32.0, 128.0, 0.0],
                [0.0, 0.0, 32.0, 128.0, 45.0],
                [4.0, 0.0, 64.0, 16.0, 0.0],  # 4.0 == 0.0 + STRIDE (4)
                [4.0, 0.0, 64.0, 16.0, 45.0],
                [4.0, 0.0, 32.0, 32.0, 0.0],
                [4.0, 0.0, 32.0, 32.0, 45.0],
                [4.0, 0.0, 16.0, 64.0, 0.0],
                [4.0, 0.0, 16.0, 64.0, 45.0],
                [4.0, 0.0, 128.0, 32.0, 0.0],
                [4.0, 0.0, 128.0, 32.0, 45.0],
                [4.0, 0.0, 64.0, 64.0, 0.0],
                [4.0, 0.0, 64.0, 64.0, 45.0],
                [4.0, 0.0, 32.0, 128.0, 0.0],
                [4.0, 0.0, 32.0, 128.0, 45.0],
            ]
        )

        assert torch.allclose(anchors[0].tensor, expected_anchor_tensor)


if __name__ == "__main__":
    unittest.main()
