# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import io
import struct
import torch

from detectron2.modeling import meta_arch
from detectron2.structures import ImageList

import mock

from .c10 import Caffe2Compatible
from .patcher import ROIHeadsPatcher, patch_generalized_rcnn
from .shared import alias, check_set_pb_arg, mock_torch_nn_functional_interpolate


def _cast_to_f32(f64):
    return struct.unpack("f", struct.pack("f", f64))[0]


def set_caffe2_compatible_tensor_mode(model, enable=True):
    def _fn(m):
        if isinstance(m, Caffe2Compatible):
            m.tensor_mode = enable

    model.apply(_fn)


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    batched_inputs is a list of dicts, each dict has fileds like image,
    height, width, image_id, etc ...
    # In D2, image is as 3D (C, H, W) tensor, all fields are not batched

    This function turn D2 format input to a tuple of Tensors
    """

    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # NOTE: The scale inside im_info is kept as convention and for providing
        # post-processing information if further processing is needed. For
        # current Caffe2 model definitions that don't include post-processing inside
        # the model, this number is not used.
        # NOTE: There can be a slight difference between width and height
        # scales, using a single number can results in numerical difference
        # compared with D2's post-processing.
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)


def caffe2_preprocess_image(self, inputs):
    """
    Override original preprocess_image, which is called inside the forward.
    Normalize, pad and batch the input images.
    """
    data, im_info = inputs
    data = alias(data, "data")
    im_info = alias(im_info, "im_info")
    normalized_data = self.normalizer(data)
    normalized_data = alias(normalized_data, "normalized_data")

    # Pack (data, im_info) into ImageList which is recognized by self.inference.
    images = ImageList(tensor=normalized_data, image_sizes=im_info)

    return images


@contextlib.contextmanager
def mock_preprocess_image(instance):
    with mock.patch.object(
        type(instance), "preprocess_image", autospec=True, side_effect=caffe2_preprocess_image
    ) as mocked_func:
        yield
    assert mocked_func.call_count > 0


class Caffe2GeneralizedRCNN(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        """
        Note: it modifies torch_model in-place.
        """
        super(Caffe2GeneralizedRCNN, self).__init__()
        assert isinstance(torch_model, meta_arch.GeneralizedRCNN)
        self._wrapped_model = patch_generalized_rcnn(torch_model)
        self.eval()
        # self.tensor_mode = False
        set_caffe2_compatible_tensor_mode(self, True)

        self.roi_heads_patcher = ROIHeadsPatcher(cfg, self._wrapped_model.roi_heads)

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"GeneralizedRCNN")
        # NOTE: maybe just encode the entire cfg.MODEL

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        if not self.tensor_mode:
            return self._wrapped_model.inference(inputs)

        with mock_preprocess_image(self._wrapped_model):
            with self.roi_heads_patcher.mock_roi_heads(self.tensor_mode):
                results = self._wrapped_model.inference(inputs, do_postprocess=False)
        return tuple(results[0].flatten())


class Caffe2PanopticFPN(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        super(Caffe2PanopticFPN, self).__init__()
        assert isinstance(torch_model, meta_arch.PanopticFPN)
        self._wrapped_model = patch_generalized_rcnn(torch_model)
        self.eval()
        set_caffe2_compatible_tensor_mode(self, True)

        self.roi_heads_patcher = ROIHeadsPatcher(cfg, self._wrapped_model.roi_heads)

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"PanopticFPN")

        # Inference parameters:
        check_set_pb_arg(predict_net, "combine_on", "i", self._wrapped_model.combine_on)
        check_set_pb_arg(
            predict_net,
            "combine_overlap_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.combine_overlap_threshold),
        )
        check_set_pb_arg(
            predict_net,
            "combine_stuff_area_limit",
            "i",
            self._wrapped_model.combine_stuff_area_limit,
        )
        check_set_pb_arg(
            predict_net,
            "combine_instances_confidence_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.combine_instances_confidence_threshold),
        )

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        """
        Re-write the inference-only forward pass of PanopticFPN in c2 style
        """
        assert self.tensor_mode

        images = caffe2_preprocess_image(self._wrapped_model, inputs)
        features = self._wrapped_model.backbone(images.tensor)

        gt_sem_seg = None
        sem_seg_results, _ = self._wrapped_model.sem_seg_head(features, gt_sem_seg)
        sem_seg_results = alias(sem_seg_results, "sem_seg")

        gt_instances = None
        proposals, _ = self._wrapped_model.proposal_generator(images, features, gt_instances)

        with self.roi_heads_patcher.mock_roi_heads(self.tensor_mode):
            detector_results, _ = self._wrapped_model.roi_heads(
                images, features, proposals, gt_instances
            )

        return tuple(detector_results[0].flatten()) + (sem_seg_results,)


class Caffe2RetinaNet(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        super(Caffe2RetinaNet, self).__init__()
        assert isinstance(torch_model, meta_arch.RetinaNet)
        self._wrapped_model = torch_model
        self.eval()
        set_caffe2_compatible_tensor_mode(self, True)

        # serialize anchor_generator for future use
        self._serialized_anchor_generator = io.BytesIO()
        torch.save(self._wrapped_model.anchor_generator, self._serialized_anchor_generator)

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"RetinaNet")

        # Inference parameters:
        check_set_pb_arg(
            predict_net, "score_threshold", "f", _cast_to_f32(self._wrapped_model.score_threshold)
        )
        check_set_pb_arg(predict_net, "topk_candidates", "i", self._wrapped_model.topk_candidates)
        check_set_pb_arg(
            predict_net, "nms_threshold", "f", _cast_to_f32(self._wrapped_model.nms_threshold)
        )
        check_set_pb_arg(
            predict_net,
            "max_detections_per_image",
            "i",
            self._wrapped_model.max_detections_per_image,
        )

        check_set_pb_arg(
            predict_net,
            "bbox_reg_weights",
            "floats",
            [_cast_to_f32(w) for w in self._wrapped_model.box2box_transform.weights],
        )
        self._encode_anchor_generator_cfg(predict_net)

    def _encode_anchor_generator_cfg(self, predict_net):
        # Ideally we can put anchor generating inside the model, then we don't
        # need to store this information.
        bytes = self._serialized_anchor_generator.getvalue()
        check_set_pb_arg(predict_net, "serialized_anchor_generator", "s", bytes)

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        assert self.tensor_mode
        images = caffe2_preprocess_image(self._wrapped_model, inputs)

        # explicitly return the images sizes to avoid removing "im_info" by ONNX
        # since it's not used in the forward path
        return_tensors = [images.image_sizes]

        features = self._wrapped_model.backbone(images.tensor)
        features = [features[f] for f in self._wrapped_model.in_features]
        for i, feature_i in enumerate(features):
            features[i] = alias(feature_i, "feature_{}".format(i), is_backward=True)
            return_tensors.append(features[i])

        box_cls, box_delta = self._wrapped_model.head(features)
        for i, (box_cls_i, box_delta_i) in enumerate(zip(box_cls, box_delta)):
            return_tensors.append(alias(box_cls_i, "box_cls_{}".format(i)))
            return_tensors.append(alias(box_delta_i, "box_delta_{}".format(i)))

        return tuple(return_tensors)


META_ARCH_CAFFE2_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": Caffe2GeneralizedRCNN,
    "PanopticFPN": Caffe2PanopticFPN,
    "RetinaNet": Caffe2RetinaNet,
}
