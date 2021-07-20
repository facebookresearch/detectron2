# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
import tempfile
import unittest
import torch
from torch import Tensor, nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.instantiate import dump_dataclass, instantiate
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.export.flatten import TracingAdapter, flatten_to_tuple
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import (
    assert_instances_allclose,
    convert_scripted_instances,
    get_sample_coco_image,
    random_boxes,
)


"""
https://detectron2.readthedocs.io/tutorials/deployment.html
contains some explanations of this file.
"""


@unittest.skipIf(os.environ.get("CI") or TORCH_VERSION < (1, 8), "Insufficient Pytorch version")
class TestScripting(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNN(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_rcnn_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testRetinaNet(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_retinanet_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    def _test_rcnn_model(self, config_path):
        model = model_zoo.get(config_path, trained=True)
        model.eval()

        fields = {
            "proposal_boxes": Boxes,
            "objectness_logits": Tensor,
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
            "pred_masks": Tensor,
        }
        script_model = scripting_with_instances(model, fields)

        inputs = [{"image": get_sample_coco_image()}] * 2
        with torch.no_grad():
            instance = model.inference(inputs, do_postprocess=False)[0]
            scripted_instance = script_model.inference(inputs, do_postprocess=False)[0]
        assert_instances_allclose(instance, scripted_instance)

    def _test_retinanet_model(self, config_path):
        model = model_zoo.get(config_path, trained=True)
        model.eval()

        fields = {
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
        }
        script_model = scripting_with_instances(model, fields)

        img = get_sample_coco_image()
        inputs = [{"image": img}] * 2
        with torch.no_grad():
            instance = model(inputs)[0]["instances"]
            scripted_instance = convert_scripted_instances(script_model(inputs)[0])
            scripted_instance = detector_postprocess(scripted_instance, img.shape[1], img.shape[2])
        assert_instances_allclose(instance, scripted_instance)
        # Note that the model currently cannot be saved and loaded into a new process:
        # https://github.com/pytorch/pytorch/issues/46944


@unittest.skipIf(os.environ.get("CI") or TORCH_VERSION < (1, 8), "Insufficient Pytorch version")
class TestTracing(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNN(self):
        # TODO: this test requires manifold access, see: T88318502
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testRetinaNet(self):
        # TODO: this test requires manifold access, see: T88318502
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func)

    def _test_model(self, config_path, inference_func):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()

        wrapper = TracingAdapter(model, image, inference_func)
        wrapper.eval()
        with torch.no_grad():
            small_image = nn.functional.interpolate(image, scale_factor=0.5)
            # trace with a different image, and the trace must still work
            traced_model = torch.jit.trace(wrapper, (small_image,))

            output = inference_func(model, image)
            traced_output = wrapper.outputs_schema(traced_model(image))
        assert_instances_allclose(output, traced_output, size_as_tensor=True)

    def testKeypointHead(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = KRCNNConvDeconvUpsampleHead(
                    ShapeSpec(channels=4, height=14, width=14), num_keypoints=17, conv_dims=(4,)
                )

            def forward(self, x, predbox1, predbox2):
                inst = [
                    Instances((100, 100), pred_boxes=Boxes(predbox1)),
                    Instances((100, 100), pred_boxes=Boxes(predbox2)),
                ]
                ret = self.model(x, inst)
                return tuple(x.pred_keypoints for x in ret)

        model = M()
        model.eval()

        def gen_input(num1, num2):
            feat = torch.randn((num1 + num2, 4, 14, 14))
            box1 = random_boxes(num1)
            box2 = random_boxes(num2)
            return feat, box1, box2

        with torch.no_grad(), patch_builtin_len():
            trace = torch.jit.trace(model, gen_input(15, 15), check_trace=False)

            inputs = gen_input(12, 10)
            trace_outputs = trace(*inputs)
            true_outputs = model(*inputs)
            for trace_output, true_output in zip(trace_outputs, true_outputs):
                self.assertTrue(torch.allclose(trace_output, true_output))


class TestTorchscriptUtils(unittest.TestCase):
    # TODO: add test to dump scripting
    def test_dump_IR_tracing(self):
        cfg = get_cfg()
        cfg.MODEL.RESNETS.DEPTH = 18
        cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

        class Mod(nn.Module):
            def forward(self, x):
                return tuple(self.m(x).values())

        model = Mod()
        model.m = build_backbone(cfg)
        model.eval()

        with torch.no_grad():
            ts_model = torch.jit.trace(model, (torch.rand(2, 3, 224, 224),))

        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            dump_torchscript_IR(ts_model, d)
            # check that the files are created
            for name in ["model_ts_code", "model_ts_IR", "model_ts_IR_inlined", "model"]:
                fname = os.path.join(d, name + ".txt")
                self.assertTrue(os.stat(fname).st_size > 0, fname)

    def test_dump_IR_function(self):
        @torch.jit.script
        def gunc(x, y):
            return x + y

        def func(x, y):
            return x + y + gunc(x, y)

        ts_model = torch.jit.trace(func, (torch.rand(3), torch.rand(3)))
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            dump_torchscript_IR(ts_model, d)
            for name in ["model_ts_code", "model_ts_IR", "model_ts_IR_inlined"]:
                fname = os.path.join(d, name + ".txt")
                self.assertTrue(os.stat(fname).st_size > 0, fname)

    def test_flatten_basic(self):
        obj = [3, ([5, 6], {"name": [7, 9], "name2": 3})]
        res, schema = flatten_to_tuple(obj)
        self.assertEqual(res, (3, 5, 6, 7, 9, 3))
        new_obj = schema(res)
        self.assertEqual(new_obj, obj)

        _, new_schema = flatten_to_tuple(new_obj)
        self.assertEqual(schema, new_schema)  # test __eq__
        self._check_schema(schema)

    def _check_schema(self, schema):
        dumped_schema = dump_dataclass(schema)
        # Check that the schema is json-serializable
        # Although in reality you might want to use yaml because it often has many levels
        json.dumps(dumped_schema)

        # Check that the schema can be deserialized
        new_schema = instantiate(dumped_schema)
        self.assertEqual(schema, new_schema)

    def test_flatten_instances_boxes(self):
        inst = Instances(
            torch.tensor([5, 8]), pred_masks=torch.tensor([3]), pred_boxes=Boxes(torch.ones((1, 4)))
        )
        obj = [3, ([5, 6], inst)]
        res, schema = flatten_to_tuple(obj)
        self.assertEqual(res[:3], (3, 5, 6))
        for r, expected in zip(res[3:], (inst.pred_boxes.tensor, inst.pred_masks, inst.image_size)):
            self.assertIs(r, expected)
        new_obj = schema(res)
        assert_instances_allclose(new_obj[1][1], inst, rtol=0.0, size_as_tensor=True)

        self._check_schema(schema)

    def test_allow_non_tensor(self):
        data = (torch.tensor([5, 8]), 3)  # contains non-tensor

        class M(nn.Module):
            def forward(self, input, number):
                return input

        model = M()
        with self.assertRaisesRegex(ValueError, "must only contain tensors"):
            adap = TracingAdapter(model, data, allow_non_tensor=False)

        adap = TracingAdapter(model, data, allow_non_tensor=True)
        _ = adap(*adap.flattened_inputs)

        newdata = (data[0].clone(),)
        with self.assertRaisesRegex(ValueError, "cannot generalize"):
            _ = adap(*newdata)
