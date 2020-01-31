# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import unittest
import torch

import detectron2.model_zoo as model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.events import EventStorage


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


def create_model_input(img, inst=None):
    if inst is not None:
        return {"image": img, "instances": inst}
    else:
        return {"image": img}


def get_empty_instance(h, w):
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.rand(0, 4))
    inst.gt_classes = torch.tensor([]).to(dtype=torch.int64)
    inst.gt_masks = BitMasks(torch.rand(0, h, w))
    return inst


def get_regular_bitmask_instances(h, w):
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.rand(3, 4))
    inst.gt_boxes.tensor[:, 2:] += inst.gt_boxes.tensor[:, :2]
    inst.gt_classes = torch.tensor([3, 4, 5]).to(dtype=torch.int64)
    inst.gt_masks = BitMasks((torch.rand(3, h, w) > 0.5))
    return inst


class ModelE2ETest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo(self.CONFIG_PATH)

    def _test_eval(self, input_sizes):
        inputs = [create_model_input(torch.rand(3, s[0], s[1])) for s in input_sizes]
        self.model.eval()
        self.model(inputs)

    def _test_train(self, input_sizes, instances):
        assert len(input_sizes) == len(instances)
        inputs = [
            create_model_input(torch.rand(3, s[0], s[1]), inst)
            for s, inst in zip(input_sizes, instances)
        ]
        self.model.train()
        with EventStorage():
            losses = self.model(inputs)
            sum(losses.values()).backward()
            del losses


class MaskRCNNE2ETest(ModelE2ETest):
    CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    def test_empty_data(self):
        instances = [get_empty_instance(200, 250), get_empty_instance(200, 249)]
        self._test_eval([(200, 250), (200, 249)])
        self._test_train([(200, 250), (200, 249)], instances)

    def test_half_empty_data(self):
        instances = [get_empty_instance(200, 250), get_regular_bitmask_instances(200, 249)]
        self._test_train([(200, 250), (200, 249)], instances)


class RetinaNetE2ETest(ModelE2ETest):
    CONFIG_PATH = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"

    def test_empty_data(self):
        instances = [get_empty_instance(200, 250), get_empty_instance(200, 249)]
        self._test_eval([(200, 250), (200, 249)])
        self._test_train([(200, 250), (200, 249)], instances)
