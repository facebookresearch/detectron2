# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import torch

from detectron2.structures import BitMasks, Boxes, Instances

from .common import get_model


# TODO(plabatut): Modularize detectron2 tests and re-use
def make_model_inputs(image, instances=None):
    if instances is None:
        return {"image": image}

    return {"image": image, "instances": instances}


def make_empty_instances(h, w):
    instances = Instances((h, w))
    instances.gt_boxes = Boxes(torch.rand(0, 4))
    instances.gt_classes = torch.tensor([]).to(dtype=torch.int64)
    instances.gt_masks = BitMasks(torch.rand(0, h, w))
    return instances


class ModelE2ETest(unittest.TestCase):
    CONFIG_PATH = ""

    def setUp(self):
        self.model = get_model(self.CONFIG_PATH)

    def _test_eval(self, sizes):
        inputs = [make_model_inputs(torch.rand(3, size[0], size[1])) for size in sizes]
        self.model.eval()
        self.model(inputs)


class DensePoseRCNNE2ETest(ModelE2ETest):
    CONFIG_PATH = "densepose_rcnn_R_101_FPN_s1x.yaml"

    def test_empty_data(self):
        self._test_eval([(200, 250), (200, 249)])
