# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .base import RectangleVisualizer, TextVisualizer


class BoundingBoxVisualizer(object):
    def __init__(self):
        self.rectangle_visualizer = RectangleVisualizer()

    def visualize(self, image_bgr, boxes_xywh):
        for bbox_xywh in boxes_xywh:
            image_bgr = self.rectangle_visualizer.visualize(image_bgr, bbox_xywh)
        return image_bgr


class ScoredBoundingBoxVisualizer(object):
    def __init__(self, bbox_visualizer_params=None, score_visualizer_params=None):
        if bbox_visualizer_params is None:
            bbox_visualizer_params = {}
        if score_visualizer_params is None:
            score_visualizer_params = {}
        self.visualizer_bbox = RectangleVisualizer(**bbox_visualizer_params)
        self.visualizer_score = TextVisualizer(**score_visualizer_params)

    def visualize(self, image_bgr, scored_bboxes):
        boxes_xywh, box_scores = scored_bboxes
        assert len(boxes_xywh) == len(
            box_scores
        ), "Number of bounding boxes {} should be equal to the number of scores {}".format(
            len(boxes_xywh), len(box_scores)
        )
        for i, box_xywh in enumerate(boxes_xywh):
            score_i = box_scores[i]
            image_bgr = self.visualizer_bbox.visualize(image_bgr, box_xywh)
            score_txt = "{0:6.4f}".format(score_i)
            topleft_xy = box_xywh[0], box_xywh[1]
            image_bgr = self.visualizer_score.visualize(image_bgr, score_txt, topleft_xy)
        return image_bgr
