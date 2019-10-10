# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File:

import numpy as np
import unittest
import torch

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def _random_data(self):
        H, W = 100, 100
        N = 10
        img = np.random.rand(H, W, 3) * 255
        boxxy = np.random.rand(N, 2) * (H // 2)
        boxes = np.concatenate((boxxy, boxxy + H // 2), axis=1)

        def _rand_poly():
            return np.random.rand(3, 2).flatten() * H

        polygons = [[_rand_poly() for _ in range(np.random.randint(1, 5))] for _ in range(N)]

        mask = np.zeros_like(img[:, :, 0], dtype=np.bool)
        mask[:10, 10:20] = 1

        labels = [str(i) for i in range(N)]
        return img, boxes, labels, polygons, [mask] * N

    @property
    def metadata(self):
        return MetadataCatalog.get("coco_2017_train")

    def test_overlay_instances(self):
        img, boxes, labels, polygons, masks = self._random_data()

        v = Visualizer(img, self.metadata)
        output = v.overlay_instances(masks=polygons, boxes=boxes, labels=labels).get_image()
        self.assertEqual(output.shape, img.shape)

        # Test 2x scaling
        v = Visualizer(img, self.metadata, scale=2.0)
        output = v.overlay_instances(masks=polygons, boxes=boxes, labels=labels).get_image()
        self.assertEqual(output.shape[0], img.shape[0] * 2)

        # Test overlay masks
        v = Visualizer(img, self.metadata)
        output = v.overlay_instances(masks=masks, boxes=boxes, labels=labels).get_image()
        self.assertEqual(output.shape, img.shape)

    def test_overlay_instances_no_boxes(self):
        img, boxes, labels, polygons, _ = self._random_data()
        v = Visualizer(img, self.metadata)
        v.overlay_instances(masks=polygons, boxes=None, labels=labels).get_image()

    def test_draw_instance_predictions(self):
        img, boxes, _, _, masks = self._random_data()
        num_inst = len(boxes)
        inst = Instances((img.shape[0], img.shape[1]))
        inst.pred_classes = torch.randint(0, 80, size=(num_inst,))
        inst.scores = torch.rand(num_inst)
        inst.pred_boxes = torch.from_numpy(boxes)
        inst.pred_masks = torch.from_numpy(np.asarray(masks))

        v = Visualizer(img, self.metadata)
        v.draw_instance_predictions(inst)

    def test_correct_output_shape(self):
        img = np.random.rand(928, 928, 3) * 255
        v = Visualizer(img, self.metadata)
        out = v.output.get_image()
        self.assertEqual(out.shape, img.shape)
