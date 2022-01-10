# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import torch
import numpy as np

from typing import Dict
from detectron2.structures import Boxes, Instances
from detectron2.config import instantiate, CfgNode as CfgNode_
from detectron2.tracking.base_tracker import build_tracker_head
from detectron2.tracking.bbox_iou_tracker import BBoxIOUTracker  # noqa
from copy import deepcopy


class TestBBoxIOUTracker(unittest.TestCase):
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
            "_target_": "detectron2.tracking.bbox_iou_tracker.BBoxIOUTracker",
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

    def test_from_config(self):
        cfg = CfgNode_()
        cfg.TRACKER_HEADS = CfgNode_()
        cfg.TRACKER_HEADS.TRACKER_NAME = "BBoxIOUTracker"
        cfg.TRACKER_HEADS.VIDEO_HEIGHT = int(self._img_size[0])
        cfg.TRACKER_HEADS.VIDEO_WIDTH = int(self._img_size[1])
        cfg.TRACKER_HEADS.MAX_NUM_INSTANCES = self._max_num_instances
        cfg.TRACKER_HEADS.MAX_LOST_FRAME_COUNT = self._max_lost_frame_count
        cfg.TRACKER_HEADS.MIN_BOX_REL_DIM = self._min_box_rel_dim
        cfg.TRACKER_HEADS.MIN_INSTANCE_PERIOD = self._min_instance_period
        cfg.TRACKER_HEADS.TRACK_IOU_THRESHOLD = self._track_iou_threshold
        tracker = build_tracker_head(cfg)
        self.assertTrue(tracker._video_height == self._img_size[0])

    def test_initialize_extra_fields(self):
        cfg = {
            "_target_": "detectron2.tracking.bbox_iou_tracker.BBoxIOUTracker",
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

    def test_assign_new_id(self):
        cfg = {
            "_target_": "detectron2.tracking.bbox_iou_tracker.BBoxIOUTracker",
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold
        }
        tracker = instantiate(cfg)
        instances = deepcopy(self._curr_instances)
        instances = tracker._initialize_extra_fields(instances)
        instances = tracker._assign_new_id(instances)
        self.assertTrue(len(instances.ID) == 2)
        self.assertTrue(instances.ID[0] == 2)
        self.assertTrue(instances.ID[1] == 3)

    def test_update(self):
        cfg = {
            "_target_": "detectron2.tracking.bbox_iou_tracker.BBoxIOUTracker",
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold
        }
        tracker = instantiate(cfg)
        prev_instances = tracker.update(self._prev_instances)
        self.assertTrue(len(prev_instances.ID) == 2)
        self.assertTrue(prev_instances.ID[0] == 0)
        self.assertTrue(prev_instances.ID[1] == 1)
        curr_instances = tracker.update(self._curr_instances)
        self.assertTrue(len(curr_instances.ID) == 2)
        self.assertTrue(curr_instances.ID[0] == 1)
        self.assertTrue(curr_instances.ID[1] == 0)


if __name__ == "__main__":
    unittest.main()
