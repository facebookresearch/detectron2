# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import unittest
import torch

import detectron2.model_zoo as model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.analysis import flop_count_operators


def get_model_zoo(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg_file = model_zoo.get_config_file(config_path)
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return build_model(cfg)


class RetinaNetFlopTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    def test_flop(self):
        # RetinaNet supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)
        self.assertTrue(int(res["conv"]), 146)  # 146B flops


class FasterRCNNFlopTest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    def test_flop(self):
        # Faster R-CNN supports flop-counting with random inputs
        inputs = [{"image": torch.rand(3, 800, 800)}]
        res = flop_count_operators(self.model, inputs)

        # This only checks flops for backbone & proposal generator
        # Flops for box head is not conv, and depends on #proposals, which is
        # almost 0 for random inputs.
        self.assertTrue(int(res["conv"]), 117)
