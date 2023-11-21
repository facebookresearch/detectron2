# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import List, Optional, Sequence, Tuple
import torch

from detectron2.layers.nms import batched_nms
from detectron2.structures.instances import Instances

from densepose.converters import ToChartResultConverterWithConfidences
from densepose.structures import (
    DensePoseChartResultWithConfidences,
    DensePoseEmbeddingPredictorOutput,
)
from densepose.vis.bounding_box import BoundingBoxVisualizer, ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import DensePoseOutputsVertexVisualizer
from densepose.vis.densepose_results import DensePoseResultsVisualizer

from .base import CompoundVisualizer

Scores = Sequence[float]
DensePoseChartResultsWithConfidences = List[DensePoseChartResultWithConfidences]


def extract_scores_from_instances(instances: Instances, select=None):
    if instances.has("scores"):
        return instances.scores if select is None else instances.scores[select]
    return None


def extract_boxes_xywh_from_instances(instances: Instances, select=None):
    if instances.has("pred_boxes"):
        boxes_xywh = instances.pred_boxes.tensor.clone()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        return boxes_xywh if select is None else boxes_xywh[select]
    return None


def create_extractor(visualizer: object):
    """
    Create an extractor for the provided visualizer
    """
    if isinstance(visualizer, CompoundVisualizer):
        extractors = [create_extractor(v) for v in visualizer.visualizers]
        return CompoundExtractor(extractors)
    elif isinstance(visualizer, DensePoseResultsVisualizer):
        return DensePoseResultExtractor()
    elif isinstance(visualizer, ScoredBoundingBoxVisualizer):
        return CompoundExtractor([extract_boxes_xywh_from_instances, extract_scores_from_instances])
    elif isinstance(visualizer, BoundingBoxVisualizer):
        return extract_boxes_xywh_from_instances
    elif isinstance(visualizer, DensePoseOutputsVertexVisualizer):
        return DensePoseOutputsExtractor()
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"Could not create extractor for {visualizer}")
        return None


class BoundingBoxExtractor:
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances):
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        return boxes_xywh


class ScoredBoundingBoxExtractor:
    """
    Extracts bounding boxes from instances
    """

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if (scores is None) or (boxes_xywh is None):
            return (boxes_xywh, scores)
        if select is not None:
            scores = scores[select]
            boxes_xywh = boxes_xywh[select]
        return (boxes_xywh, scores)


class DensePoseResultExtractor:
    """
    Extracts DensePose chart result with confidences from instances
    """

    def __call__(
        self, instances: Instances, select=None
    ) -> Tuple[Optional[DensePoseChartResultsWithConfidences], Optional[torch.Tensor]]:
        if instances.has("pred_densepose") and instances.has("pred_boxes"):
            dpout = instances.pred_densepose
            boxes_xyxy = instances.pred_boxes
            boxes_xywh = extract_boxes_xywh_from_instances(instances)
            if select is not None:
                dpout = dpout[select]
                boxes_xyxy = boxes_xyxy[select]
            converter = ToChartResultConverterWithConfidences()
            results = [converter.convert(dpout[i], boxes_xyxy[[i]]) for i in range(len(dpout))]
            return results, boxes_xywh
        else:
            return None, None


class DensePoseOutputsExtractor:
    """
    Extracts DensePose result from instances
    """

    def __call__(
        self,
        instances: Instances,
        select=None,
    ) -> Tuple[
        Optional[DensePoseEmbeddingPredictorOutput], Optional[torch.Tensor], Optional[List[int]]
    ]:
        if not (instances.has("pred_densepose") and instances.has("pred_boxes")):
            return None, None, None

        dpout = instances.pred_densepose
        boxes_xyxy = instances.pred_boxes
        boxes_xywh = extract_boxes_xywh_from_instances(instances)

        if instances.has("pred_classes"):
            classes = instances.pred_classes.tolist()
        else:
            classes = None

        if select is not None:
            dpout = dpout[select]
            boxes_xyxy = boxes_xyxy[select]
            if classes is not None:
                classes = classes[select]

        return dpout, boxes_xywh, classes


class CompoundExtractor:
    """
    Extracts data for CompoundVisualizer
    """

    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self, instances: Instances, select=None):
        datas = []
        for extractor in self.extractors:
            data = extractor(instances, select)
            datas.append(data)
        return datas


class NmsFilteredExtractor:
    """
    Extracts data in the format accepted by NmsFilteredVisualizer
    """

    def __init__(self, extractor, iou_threshold):
        self.extractor = extractor
        self.iou_threshold = iou_threshold

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        if boxes_xywh is None:
            return None
        select_local_idx = batched_nms(
            boxes_xywh,
            scores,
            torch.zeros(len(scores), dtype=torch.int32),
            iou_threshold=self.iou_threshold,
        ).squeeze()
        select_local = torch.zeros(len(boxes_xywh), dtype=torch.bool, device=boxes_xywh.device)
        select_local[select_local_idx] = True
        select = select_local if select is None else (select & select_local)
        return self.extractor(instances, select=select)


class ScoreThresholdedExtractor:
    """
    Extracts data in the format accepted by ScoreThresholdedVisualizer
    """

    def __init__(self, extractor, min_score):
        self.extractor = extractor
        self.min_score = min_score

    def __call__(self, instances: Instances, select=None):
        scores = extract_scores_from_instances(instances)
        if scores is None:
            return None
        select_local = scores > self.min_score
        select = select_local if select is None else (select & select_local)
        data = self.extractor(instances, select=select)
        return data
