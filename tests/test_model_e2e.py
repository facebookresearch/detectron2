# -*- coding: utf-8 -*-


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


class MaskRCNNE2ETest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

    def test_empty_data(self):
        inst = [get_empty_instance(200, 250), get_empty_instance(200, 249)]

        # eval
        self.model.eval()
        self.model(
            [
                create_model_input(torch.rand(3, 200, 250)),
                create_model_input(torch.rand(3, 200, 249)),
            ]
        )

        # training
        self.model.train()
        with EventStorage():
            losses = self.model(
                [
                    create_model_input(torch.rand(3, 200, 250), inst[0]),
                    create_model_input(torch.rand(3, 200, 249), inst[1]),
                ]
            )
            sum(losses.values()).backward()
            del losses


class RetinaNetE2ETest(unittest.TestCase):
    def setUp(self):
        self.model = get_model_zoo("COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    def test_empty_data(self):
        inst = [get_empty_instance(200, 250), get_empty_instance(200, 249)]

        # eval
        self.model.eval()
        self.model(
            [
                create_model_input(torch.rand(3, 200, 250)),
                create_model_input(torch.rand(3, 200, 249)),
            ]
        )

        # training
        self.model.train()
        with EventStorage():
            losses = self.model(
                [
                    create_model_input(torch.rand(3, 200, 250), inst[0]),
                    create_model_input(torch.rand(3, 200, 249), inst[1]),
                ]
            )
            sum(losses.values()).backward()
            del losses
