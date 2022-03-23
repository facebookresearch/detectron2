# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import fields
from typing import Any, List
import torch

from detectron2.structures import Instances
from densepose.converters.segm_to_mask import get_probability_mean


def densepose_inference(densepose_predictor_output: Any, detections: List[Instances], score=None):
    """
    Splits DensePose predictor outputs into chunks, each chunk corresponds to
    detections on one image. Predictor output chunks are stored in `pred_densepose`
    attribute of the corresponding `Instances` object.

    Args:
        densepose_predictor_output: a dataclass instance (can be of different types,
            depending on predictor used for inference). Each field can be `None`
            (if the corresponding output was not inferred) or a tensor of size
            [N, ...], where N = N_1 + N_2 + .. + N_k is a total number of
            detections on all images, N_1 is the number of detections on image 1,
            N_2 is the number of detections on image 2, etc.
        detections: a list of objects of type `Instance`, k-th object corresponds
            to detections on k-th image.
    """
    k = 0

    """
    Score part: get pred scores
    """
    if score is not None:
        if len(score) == 1:
            pred_scores = score[0].view(-1)
        elif len(score) == 2:
            pred_scores = torch.sqrt(score[0].view(-1) * score[1][0].view(-1))

    for detection_i in detections:
        if densepose_predictor_output is None:
            # don't add `pred_densepose` attribute
            continue
        n_i = len(detection_i)
        PredictorOutput = type(densepose_predictor_output)
        output_i_dict = {}
        # we assume here that `densepose_predictor_output` is a dataclass object
        for field in fields(densepose_predictor_output):
            field_value = getattr(densepose_predictor_output, field.name)
            # slice tensors
            if isinstance(field_value, torch.Tensor):
                output_i_dict[field.name] = field_value[k : k + n_i]
            # leave others as is
            else:
                output_i_dict[field.name] = field_value
        detection_i.pred_densepose = PredictorOutput(**output_i_dict)

        """
        Score part: compute socre for each proposals
        """
        if score is not None:
            mean = get_probability_mean(detection_i.pred_densepose, detection_i.pred_boxes, detection_i.image_size)
            detection_i.scores = torch.pow(torch.pow(detection_i.scores * pred_scores[k: k + n_i], 2) * mean, 1/5)
            # detection_i.scores = torch.pow(detection_i.scores * pred_scores[k: k + n_i], 1/2)


        k += n_i
