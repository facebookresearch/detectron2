# Copyright (c) Facebook, Inc. and its affiliates.


import unittest
import torch

from detectron2.utils.analysis import flop_count_operators, parameter_count
from detectron2.utils.testing import get_model_no_weights


class RetinaNetTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_no_weights("COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    def test_flop(self):
        # RetinaNet supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800), "test_unused": "abcd"}]
        res = flop_count_operators(self.model, inputs)
        self.assertTrue(int(res["conv"]), 146)  # 146B flops

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 37915572)
        self.assertTrue(res["backbone"], 31452352)


class FasterRCNNTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_no_weights("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    def test_flop(self):
        # Faster R-CNN supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)

        # This only checks flops for backbone & proposal generator
        # Flops for box head is not conv, and depends on #proposals, which is
        # almost 0 for random inputs.
        self.assertTrue(int(res["conv"]), 117)

    def test_param_count(self):
        res = parameter_count(self.model)
        self.assertTrue(res[""], 41699936)
        self.assertTrue(res["backbone"], 26799296)
