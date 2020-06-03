# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File:

import numpy as np
import unittest
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode, Instances, RotatedBoxes
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

    def test_draw_dataset_dict(self):
        img = np.random.rand(512, 512, 3) * 255
        dic = {
            "annotations": [
                {
                    "bbox": [
                        368.9946492271106,
                        330.891438763377,
                        13.148537455410235,
                        13.644708680142685,
                    ],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "iscrowd": 1,
                    "segmentation": {
                        "counts": "_jh52m?2N2N2N2O100O10O001N1O2MceP2",
                        "size": [512, 512],
                    },
                }
            ],
            "height": 512,
            "image_id": 1,
            "width": 512,
        }
        v = Visualizer(img, self.metadata)
        v.draw_dataset_dict(dic)

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

    def test_draw_empty_mask_predictions(self):
        img, boxes, _, _, masks = self._random_data()
        num_inst = len(boxes)
        inst = Instances((img.shape[0], img.shape[1]))
        inst.pred_classes = torch.randint(0, 80, size=(num_inst,))
        inst.scores = torch.rand(num_inst)
        inst.pred_boxes = torch.from_numpy(boxes)
        inst.pred_masks = torch.from_numpy(np.zeros_like(np.asarray(masks)))

        v = Visualizer(img, self.metadata)
        v.draw_instance_predictions(inst)

    def test_correct_output_shape(self):
        img = np.random.rand(928, 928, 3) * 255
        v = Visualizer(img, self.metadata)
        out = v.output.get_image()
        self.assertEqual(out.shape, img.shape)

    def test_overlay_rotated_instances(self):
        H, W = 100, 150
        img = np.random.rand(H, W, 3) * 255
        num_boxes = 50
        boxes_5d = torch.zeros(num_boxes, 5)
        boxes_5d[:, 0] = torch.FloatTensor(num_boxes).uniform_(-0.1 * W, 1.1 * W)
        boxes_5d[:, 1] = torch.FloatTensor(num_boxes).uniform_(-0.1 * H, 1.1 * H)
        boxes_5d[:, 2] = torch.FloatTensor(num_boxes).uniform_(0, max(W, H))
        boxes_5d[:, 3] = torch.FloatTensor(num_boxes).uniform_(0, max(W, H))
        boxes_5d[:, 4] = torch.FloatTensor(num_boxes).uniform_(-1800, 1800)
        rotated_boxes = RotatedBoxes(boxes_5d)
        labels = [str(i) for i in range(num_boxes)]

        v = Visualizer(img, self.metadata)
        output = v.overlay_instances(boxes=rotated_boxes, labels=labels).get_image()
        self.assertEqual(output.shape, img.shape)

    def test_draw_no_metadata(self):
        img, boxes, _, _, masks = self._random_data()
        num_inst = len(boxes)
        inst = Instances((img.shape[0], img.shape[1]))
        inst.pred_classes = torch.randint(0, 80, size=(num_inst,))
        inst.scores = torch.rand(num_inst)
        inst.pred_boxes = torch.from_numpy(boxes)
        inst.pred_masks = torch.from_numpy(np.asarray(masks))

        v = Visualizer(img, MetadataCatalog.get("asdfasdf"))
        v.draw_instance_predictions(inst)

    def test_draw_binary_mask(self):
        img, boxes, _, _, masks = self._random_data()
        img[:, :, 0] = 0  # remove red color
        mask = masks[0]
        mask_with_hole = np.zeros_like(mask).astype("uint8")
        mask_with_hole = cv2.rectangle(mask_with_hole, (10, 10), (50, 50), 1, 5)

        for m in [mask, mask_with_hole]:
            v = Visualizer(img)
            o = v.draw_binary_mask(m, color="red", text="test")
            o = o.get_image().astype("float32")
            # red color is drawn on the image
            self.assertTrue(o[:, :, 0].sum() > 0)


if __name__ == "__main__":
    unittest.main()
