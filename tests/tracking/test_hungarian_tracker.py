# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from typing import Dict

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.structures import Boxes, Instances


class TestBaseHungarianTracker(unittest.TestCase):
    def setUp(self):
        self._img_size = np.array([600, 800])
        self._prev_boxes = np.array(
            [
                [101, 101, 200, 200],
                [301, 301, 450, 450],
            ]
        ).astype(np.float32)
        self._prev_scores = np.array([0.9, 0.9])
        self._prev_classes = np.array([1, 1])
        self._prev_masks = np.ones((2, 600, 800)).astype("uint8")
        self._curr_boxes = np.array(
            [
                [302, 303, 451, 452],
                [101, 102, 201, 203],
            ]
        ).astype(np.float32)
        self._curr_scores = np.array([0.95, 0.85])
        self._curr_classes = np.array([1, 1])
        self._curr_masks = np.ones((2, 600, 800)).astype("uint8")

        self._prev_instances = {
            "image_size": self._img_size,
            "pred_boxes": self._prev_boxes,
            "scores": self._prev_scores,
            "pred_classes": self._prev_classes,
            "pred_masks": self._prev_masks,
        }
        self._prev_instances = self._convertDictPredictionToInstance(self._prev_instances)
        self._curr_instances = {
            "image_size": self._img_size,
            "pred_boxes": self._curr_boxes,
            "scores": self._curr_scores,
            "pred_classes": self._curr_classes,
            "pred_masks": self._curr_masks,
        }
        self._curr_instances = self._convertDictPredictionToInstance(self._curr_instances)

        self._max_num_instances = 200
        self._max_lost_frame_count = 0
        self._min_box_rel_dim = 0.02
        self._min_instance_period = 1
        self._track_iou_threshold = 0.5

    def _convertDictPredictionToInstance(self, prediction: Dict) -> Instances:
        """
        convert prediction from Dict to D2 Instances format
        """
        res = Instances(
            image_size=torch.IntTensor(prediction["image_size"]),
            pred_boxes=Boxes(torch.FloatTensor(prediction["pred_boxes"])),
            pred_masks=torch.IntTensor(prediction["pred_masks"]),
            pred_classes=torch.IntTensor(prediction["pred_classes"]),
            scores=torch.FloatTensor(prediction["scores"]),
        )
        return res

    def test_init(self):
        cfg = {
            "_target_": "detectron2.tracking.hungarian_tracker.BaseHungarianTracker",
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold
        }
        tracker = instantiate(cfg)
        self.assertTrue(tracker._video_height == self._img_size[0])

    def test_initialize_extra_fields(self):
        cfg = {
            "_target_": "detectron2.tracking.hungarian_tracker.BaseHungarianTracker",
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold
        }
        tracker = instantiate(cfg)
        instances = tracker._initialize_extra_fields(self._curr_instances)
        self.assertTrue(instances.has("ID"))
        self.assertTrue(instances.has("ID_period"))
        self.assertTrue(instances.has("lost_frame_count"))


if __name__ == "__main__":
    unittest.main()
