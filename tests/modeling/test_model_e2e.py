# Copyright (c) Facebook, Inc. and its affiliates.


import itertools
import numpy as np
import unittest
from contextlib import contextmanager
from copy import deepcopy
import torch

from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage
from detectron2.utils.testing import get_model_no_weights


@contextmanager
def typecheck_hook(model, *, in_dtype=None, out_dtype=None):
    """
    Check that the model must be called with the given input/output dtype
    """
    if not isinstance(in_dtype, set):
        in_dtype = {in_dtype}
    if not isinstance(out_dtype, set):
        out_dtype = {out_dtype}

    def flatten(x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            return list(itertools.chain(*[flatten(t) for t in x]))
        if isinstance(x, dict):
            return flatten(list(x.values()))
        return []

    def hook(module, input, output):
        if in_dtype is not None:
            dtypes = {x.dtype for x in flatten(input)}
            assert (
                dtypes == in_dtype
            ), f"Expected input dtype of {type(module)} is {in_dtype}. Got {dtypes} instead!"

        if out_dtype is not None:
            dtypes = {x.dtype for x in flatten(output)}
            assert (
                dtypes == out_dtype
            ), f"Expected output dtype of {type(module)} is {out_dtype}. Got {dtypes} instead!"

    with model.register_forward_hook(hook):
        yield


def create_model_input(img, inst=None):
    if inst is not None:
        return {"image": img, "instances": inst}
    else:
        return {"image": img}


def get_empty_instance(h, w):
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.rand(0, 4))
    inst.gt_classes = torch.tensor([]).to(dtype=torch.int64)
    inst.gt_masks = BitMasks(torch.rand(0, h, w))
    return inst


def get_regular_bitmask_instances(h, w):
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.rand(3, 4))
    inst.gt_boxes.tensor[:, 2:] += inst.gt_boxes.tensor[:, :2]
    inst.gt_classes = torch.tensor([3, 4, 5]).to(dtype=torch.int64)
    inst.gt_masks = BitMasks((torch.rand(3, h, w) > 0.5))
    return inst


class ModelE2ETest:
    def setUp(self):
        torch.manual_seed(43)
        self.model = get_model_no_weights(self.CONFIG_PATH)

    def _test_eval(self, input_sizes):
        inputs = [create_model_input(torch.rand(3, s[0], s[1])) for s in input_sizes]
        self.model.eval()
        self.model(inputs)

    def _test_train(self, input_sizes, instances):
        assert len(input_sizes) == len(instances)
        inputs = [
            create_model_input(torch.rand(3, s[0], s[1]), inst)
            for s, inst in zip(input_sizes, instances)
        ]
        self.model.train()
        with EventStorage():
            losses = self.model(inputs)
            sum(losses.values()).backward()
            del losses

    def _inf_tensor(self, *shape):
        return 1.0 / torch.zeros(*shape, device=self.model.device)

    def _nan_tensor(self, *shape):
        return torch.zeros(*shape, device=self.model.device).fill_(float("nan"))

    def test_empty_data(self):
        instances = [get_empty_instance(200, 250), get_empty_instance(200, 249)]
        self._test_eval([(200, 250), (200, 249)])
        self._test_train([(200, 250), (200, 249)], instances)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_eval_tocpu(self):
        model = deepcopy(self.model).cpu()
        model.eval()
        input_sizes = [(200, 250), (200, 249)]
        inputs = [create_model_input(torch.rand(3, s[0], s[1])) for s in input_sizes]
        model(inputs)


class MaskRCNNE2ETest(ModelE2ETest, unittest.TestCase):
    CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    def test_half_empty_data(self):
        instances = [get_empty_instance(200, 250), get_regular_bitmask_instances(200, 249)]
        self._test_train([(200, 250), (200, 249)], instances)

    # This test is flaky because in some environment the output features are zero due to relu
    # def test_rpn_inf_nan_data(self):
    #     self.model.eval()
    #     for tensor in [self._inf_tensor, self._nan_tensor]:
    #         images = ImageList(tensor(1, 3, 512, 512), [(510, 510)])
    #         features = {
    #             "p2": tensor(1, 256, 256, 256),
    #             "p3": tensor(1, 256, 128, 128),
    #             "p4": tensor(1, 256, 64, 64),
    #             "p5": tensor(1, 256, 32, 32),
    #             "p6": tensor(1, 256, 16, 16),
    #         }
    #         props, _ = self.model.proposal_generator(images, features)
    #         self.assertEqual(len(props[0]), 0)

    def test_roiheads_inf_nan_data(self):
        self.model.eval()
        for tensor in [self._inf_tensor, self._nan_tensor]:
            images = ImageList(tensor(1, 3, 512, 512), [(510, 510)])
            features = {
                "p2": tensor(1, 256, 256, 256),
                "p3": tensor(1, 256, 128, 128),
                "p4": tensor(1, 256, 64, 64),
                "p5": tensor(1, 256, 32, 32),
                "p6": tensor(1, 256, 16, 16),
            }
            props = [Instances((510, 510))]
            props[0].proposal_boxes = Boxes([[10, 10, 20, 20]]).to(device=self.model.device)
            props[0].objectness_logits = torch.tensor([1.0]).reshape(1, 1)
            det, _ = self.model.roi_heads(images, features, props)
            self.assertEqual(len(det[0]), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_autocast(self):
        from torch.cuda.amp import autocast

        inputs = [{"image": torch.rand(3, 100, 100)}]
        self.model.eval()
        with autocast(), typecheck_hook(
            self.model.backbone, in_dtype=torch.float32, out_dtype=torch.float16
        ), typecheck_hook(
            self.model.roi_heads.box_predictor, in_dtype=torch.float16, out_dtype=torch.float16
        ):
            out = self.model.inference(inputs, do_postprocess=False)[0]
            self.assertEqual(out.pred_boxes.tensor.dtype, torch.float32)
            self.assertEqual(out.pred_masks.dtype, torch.float16)
            self.assertEqual(out.scores.dtype, torch.float32)  # scores comes from softmax


class RetinaNetE2ETest(ModelE2ETest, unittest.TestCase):
    CONFIG_PATH = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"

    def test_inf_nan_data(self):
        self.model.eval()
        self.model.score_threshold = -999999999
        for tensor in [self._inf_tensor, self._nan_tensor]:
            images = ImageList(tensor(1, 3, 512, 512), [(510, 510)])
            features = [
                tensor(1, 256, 128, 128),
                tensor(1, 256, 64, 64),
                tensor(1, 256, 32, 32),
                tensor(1, 256, 16, 16),
                tensor(1, 256, 8, 8),
            ]
            anchors = self.model.anchor_generator(features)
            _, pred_anchor_deltas = self.model.head(features)
            HWAs = [np.prod(x.shape[-3:]) // 4 for x in pred_anchor_deltas]

            pred_logits = [tensor(1, HWA, self.model.num_classes) for HWA in HWAs]
            pred_anchor_deltas = [tensor(1, HWA, 4) for HWA in HWAs]
            det = self.model.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            # all predictions (if any) are infinite or nan
            if len(det[0]):
                self.assertTrue(torch.isfinite(det[0].pred_boxes.tensor).sum() == 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_autocast(self):
        from torch.cuda.amp import autocast

        inputs = [{"image": torch.rand(3, 100, 100)}]
        self.model.eval()
        with autocast(), typecheck_hook(
            self.model.backbone, in_dtype=torch.float32, out_dtype=torch.float16
        ), typecheck_hook(self.model.head, in_dtype=torch.float16, out_dtype=torch.float16):
            out = self.model(inputs)[0]["instances"]
            self.assertEqual(out.pred_boxes.tensor.dtype, torch.float32)
            self.assertEqual(out.scores.dtype, torch.float16)
