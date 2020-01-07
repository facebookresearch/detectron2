# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import io
import logging
import numpy as np
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core

from detectron2.export.caffe2_modeling import convert_batched_inputs_to_c2_format
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.roi_heads import keypoint_head
from detectron2.structures import Boxes, Instances

from .shared import ScopedWS, get_pb_arg_floats, get_pb_arg_valf, get_pb_arg_vali, get_pb_arg_vals

logger = logging.getLogger(__name__)


def is_valid_model_output_blob(blob):
    return isinstance(blob, np.ndarray)


def assemble_rcnn_outputs_by_name(image_sizes, tensor_outputs, force_mask_on=False):
    """
    A function to assemble caffe2 model's outputs (i.e. Dict[str, Tensor])
    to detectron2's format (i.e. list of Instances instance).
    This only works when the model follows the Caffe2 detectron's naming convention.

    Args:
        image_sizes (List[List[int, int]]): [H, W] of every image.
        tensor_outputs (Dict[str, Tensor]): external_output to its tensor.

        force_mask_on (Bool): if true, the it make sure there'll be pred_masks even
            if the mask is not found from tensor_outputs (usually due to model crash)
    """

    results = [Instances(image_size) for image_size in image_sizes]

    batch_splits = tensor_outputs.get("batch_splits", None)
    if batch_splits:
        raise NotImplementedError()
    assert len(image_sizes) == 1
    result = results[0]

    bbox_nms = tensor_outputs["bbox_nms"]
    score_nms = tensor_outputs["score_nms"]
    class_nms = tensor_outputs["class_nms"]
    # Detection will always success because Conv support 0-batch
    assert is_valid_model_output_blob(bbox_nms)
    assert is_valid_model_output_blob(score_nms)
    assert is_valid_model_output_blob(class_nms)
    result.pred_boxes = Boxes(torch.Tensor(bbox_nms))
    result.scores = torch.Tensor(score_nms)
    result.pred_classes = torch.Tensor(class_nms).to(torch.int64)

    mask_fcn_probs = tensor_outputs.get("mask_fcn_probs", None)
    if is_valid_model_output_blob(mask_fcn_probs):
        # finish the mask pred
        mask_probs_pred = torch.Tensor(mask_fcn_probs)
        num_masks = mask_probs_pred.shape[0]
        class_pred = result.pred_classes
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]
        result.pred_masks = mask_probs_pred
    elif force_mask_on:
        # NOTE: there's no way to know the height/width of mask here, it won't be
        # used anyway when batch size is 0, so just set them to 0.
        result.pred_masks = torch.zeros([0, 1, 0, 0], dtype=torch.uint8)

    keypoints_out = tensor_outputs.get("keypoints_out", None)
    kps_score = tensor_outputs.get("kps_score", None)
    if is_valid_model_output_blob(keypoints_out):
        # keypoints_out: [N, 4, #kypoints], where 4 is in order of (x, y, score, prob)
        keypoints_tensor = torch.Tensor(keypoints_out)
        # NOTE: it's possible that prob is not calculated if "should_output_softmax"
        # is set to False in HeatmapMaxKeypoint, so just using raw score, seems
        # it doesn't affect mAP. TODO: check more carefully.
        keypoint_xyp = keypoints_tensor.transpose(1, 2)[:, :, [0, 1, 2]]
        result.pred_keypoints = keypoint_xyp
    elif is_valid_model_output_blob(kps_score):
        # keypoint heatmap to sparse data structure
        pred_keypoint_logits = torch.Tensor(kps_score)
        keypoint_head.keypoint_rcnn_inference(pred_keypoint_logits, [result])

    return results


class GeneralizedRCNNAssembler(object):
    """
    A class whose instance can assemble caffe2 model's outputs
    """

    def __init__(self):
        pass

    def post_processing(self, batched_inputs, results, image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def assemble(self, batched_inputs, c2_inputs, c2_results):
        image_sizes = [[int(im[0]), int(im[1])] for im in c2_inputs["im_info"]]
        results = assemble_rcnn_outputs_by_name(image_sizes, c2_results)
        return self.post_processing(batched_inputs, results, image_sizes)


class PanopticFPNAssembler(object):
    def __init__(
        self,
        combine_on,
        combine_overlap_threshold,
        combine_stuff_area_limit,
        combine_instances_confidence_threshold,
    ):
        self.combine_on = combine_on
        self.combine_overlap_threshold = combine_overlap_threshold
        self.combine_stuff_area_limit = combine_stuff_area_limit
        self.combine_instances_confidence_threshold = combine_instances_confidence_threshold

    def assemble(self, batched_inputs, c2_inputs, c2_results):
        image_sizes = [[int(im[0]), int(im[1])] for im in c2_inputs["im_info"]]
        detector_results = assemble_rcnn_outputs_by_name(
            image_sizes, c2_results, force_mask_on=True
        )
        sem_seg_results = torch.Tensor(c2_results["sem_seg"])

        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)

            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results


class RetinaNetAssembler(object):
    def __init__(
        self,
        anchor_generator,
        box2box_transform,
        score_threshold,
        topk_candidates,
        nms_threshold,
        max_detections_per_image,
    ):
        # num_classes cannot be determined during initialization, it'll be filled
        # before calling inference
        self.num_classes = None

        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform

        self.score_threshold = score_threshold
        self.topk_candidates = topk_candidates
        self.nms_threshold = nms_threshold
        self.max_detections_per_image = max_detections_per_image

    def inference_single_image(self, *args, **kwargs):
        return RetinaNet.inference_single_image(self, *args, **kwargs)

    def inference(self, *args, **kwargs):
        return RetinaNet.inference(self, *args, **kwargs)

    def assemble(self, batched_inputs, c2_inputs, c2_results):
        c2_results = {k: torch.Tensor(v) for k, v in c2_results.items()}

        image_sizes = [[int(im[0]), int(im[1])] for im in c2_inputs["im_info"]]

        num_features = len([x for x in c2_results.keys() if x.startswith("box_cls_")])
        box_cls = [c2_results["box_cls_{}".format(i)] for i in range(num_features)]
        box_delta = [c2_results["box_delta_{}".format(i)] for i in range(num_features)]

        # For each feature level, feature should have the same batch size and
        # spatial dimension as the box_cls and box_delta.
        dummy_features = [box_delta[i].clone()[:, 0:0, :, :] for i in range(num_features)]
        anchors = self.anchor_generator(dummy_features)

        # self.num_classess can be inferred
        self.num_classes = box_cls[0].shape[1] // (box_delta[0].shape[1] // 4)

        results = self.inference(box_cls, box_delta, anchors, image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


def best_assembler_from_predict_net(predict_net):
    meta_arch = get_pb_arg_vals(predict_net, "meta_architecture", "GeneralizedRCNN")

    if meta_arch == b"GeneralizedRCNN":
        return GeneralizedRCNNAssembler()
    elif meta_arch == b"PanopticFPN":
        return PanopticFPNAssembler(
            get_pb_arg_vali(predict_net, "combine_on", None),
            get_pb_arg_valf(predict_net, "combine_overlap_threshold", None),
            get_pb_arg_vali(predict_net, "combine_stuff_area_limit", None),
            get_pb_arg_valf(predict_net, "combine_instances_confidence_threshold", None),
        )
    elif meta_arch == b"RetinaNet":
        serialized_anchor_generator = io.BytesIO(
            get_pb_arg_vals(predict_net, "serialized_anchor_generator", None)
        )
        anchor_generator = torch.load(serialized_anchor_generator)
        bbox_reg_weights = get_pb_arg_floats(predict_net, "bbox_reg_weights", None)
        box2box_transform = Box2BoxTransform(weights=tuple(bbox_reg_weights))
        return RetinaNetAssembler(
            anchor_generator,
            box2box_transform,
            get_pb_arg_valf(predict_net, "score_threshold", None),
            get_pb_arg_vali(predict_net, "topk_candidates", None),
            get_pb_arg_valf(predict_net, "nms_threshold", None),
            get_pb_arg_vali(predict_net, "max_detections_per_image", None),
        )

    raise ValueError("Unsupported meta architecture: {}".format(meta_arch))


class ProtobufModel(torch.nn.Module):
    """
    A class works just like nn.Module in terms of inference, but running
    caffe2 model under the hood. Input/Output are Dict[str, tensor] whose keys
    are in external_input/output.
    """

    def __init__(self, predict_net, init_net):
        logger.info("Initializing ProtobufModel ...")
        super().__init__()
        assert isinstance(predict_net, caffe2_pb2.NetDef)
        assert isinstance(init_net, caffe2_pb2.NetDef)
        self.ws_name = "__ws_tmp__"
        self.net = core.Net(predict_net)

        with ScopedWS(self.ws_name, is_reset=True, is_cleanup=False) as ws:
            ws.RunNetOnce(init_net)
            for blob in self.net.Proto().external_input:
                if blob not in ws.Blobs():
                    ws.CreateBlob(blob)
            ws.CreateNet(self.net)

        self._error_msgs = set()

    def forward(self, inputs_dict):
        assert all(inp in self.net.Proto().external_input for inp in inputs_dict)
        with ScopedWS(self.ws_name, is_reset=False, is_cleanup=False) as ws:
            for b, tensor in inputs_dict.items():
                ws.FeedBlob(b, tensor)
            try:
                ws.RunNet(self.net.Proto().name)
            except RuntimeError as e:
                if not str(e) in self._error_msgs:
                    self._error_msgs.add(str(e))
                    logger.warning("Encountered new RuntimeError: \n{}".format(str(e)))
                logger.warning("Catch the error and use partial results.")

            outputs_dict = collections.OrderedDict(
                [(b, ws.FetchBlob(b)) for b in self.net.Proto().external_output]
            )
            # Remove outputs of current run, this is necessary in order to
            # prevent fetching the result from previous run if the model fails
            # in the middle.
            for b in self.net.Proto().external_output:
                # Needs to create uninitialized blob to make the net runable.
                # This is "equivalent" to: ws.RemoveBlob(b) then ws.CreateBlob(b),
                # but there'no such API.
                ws.FeedBlob(b, "{}, a C++ native class of type nullptr (uninitialized).".format(b))

        return outputs_dict


class ProtobufDetectionModel(torch.nn.Module):
    """
    A class works just like GeneralizedRCNN in terms of inference, but running
    caffe2 model under the hood.
    """

    def __init__(self, predict_net, init_net, assembler=None):
        super().__init__()
        self.protobuf_model = ProtobufModel(predict_net, init_net)

        self.size_divisibility = get_pb_arg_vali(predict_net, "size_divisibility", 0)
        self.assembler = assembler or best_assembler_from_predict_net(predict_net)

    def forward(self, batched_inputs):
        data, im_info = convert_batched_inputs_to_c2_format(
            batched_inputs, self.size_divisibility, torch.device("cpu")
        )
        c2_inputs = {"data": data, "im_info": im_info}
        c2_results = self.protobuf_model(c2_inputs)

        results = self.assembler.assemble(batched_inputs, c2_inputs, c2_results)
        return results
