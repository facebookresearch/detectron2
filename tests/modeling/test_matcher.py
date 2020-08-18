# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
import torch

from detectron2.config import get_cfg
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION


class TestMatcher(unittest.TestCase):
    # need https://github.com/pytorch/pytorch/pull/38378
    @unittest.skipIf(TORCH_VERSION < (1, 6), "Insufficient pytorch version")
    def test_scriptability(self):
        cfg = get_cfg()
        anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS,
            cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True,
            ignore_threshold=0.7,
        )

        boxes1 = Boxes(torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]))
        boxes2 = Boxes(
            torch.tensor(
                [
                    [0.0, 0.0, 2.2, 2.2],
                    [0.0, 0.0, 1.2, 2.2],
                    [0.0, 0.0, 2.2, 1.0],
                    [0.0, 0.0, 1.2, 1.2],
                    [1.1, 1.1, 2.1, 2.1],
                    [1.1, 1.1, 3.1, 3.1],
                    [0.5, 0.5, 1.4, 1.4],
                ]
            )
        )
        ignore_box = Boxes(torch.tensor([[0.5, 0.5, 1.5, 1.5]]))

        expected_matches = torch.tensor([0, 0, 0, 1, 0, 0, 0])
        expected_match_labels = torch.tensor([1, -1, -1, 1, 0, 0, -1], dtype=torch.int8)

        matches, match_labels = anchor_matcher(boxes2, boxes1, ignore_boxes=ignore_box)
        self.assertTrue(torch.allclose(matches, expected_matches))
        self.assertTrue(torch.allclose(match_labels, expected_match_labels))

        # nonzero_tuple must be import explicitly to let jit know what it is.
        # https://github.com/pytorch/pytorch/issues/38964
        # from typing import List
        # from detectron2.layers import nonzero_tuple  # noqa F401

        # def f(
        #       thresholds: List[float],
        #       labels: List[int],
        #       allow_low_quality_matches:
        #       bool, ignore_threshold: float
        #       ):
        #     return Matcher(
        #                   thresholds,
        #                   labels,
        #                   allow_low_quality_matches=True,
        #                   ignore_threshold=0.7
        #                   )

        # scripted_anchor_matcher = torch.jit.script(f)(
        #     cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS
        # )
        # matches, match_labels = scripted_anchor_matcher(boxes2, boxes1, ignore_box)
        # self.assertTrue(torch.allclose(matches, expected_matches))
        # self.assertTrue(torch.allclose(match_labels, expected_match_labels))


if __name__ == "__main__":
    unittest.main()
