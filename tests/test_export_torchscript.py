# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import glob
import json
import os
import random
import tempfile
import unittest
import zipfile
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
    skipIfOnCPUCI,
)


"""
https://detectron2.readthedocs.io/tutorials/deployment.html
contains some explanations of this file.
"""


class TestScripting(unittest.TestCase):
    def testMaskRCNNFPN(self):
        self._test_rcnn_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @skipIfOnCPUCI
    def testMaskRCNNC4(self):
        self._test_rcnn_model("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")

    def testRetinaNet(self):
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

        # Test that batch inference with different shapes are supported
        image = get_sample_coco_image()
        small_image = nn.functional.interpolate(image, scale_factor=0.5)
        inputs = [{"image": image}, {"image": small_image}]
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


# TODO: this test requires manifold access, see: T88318502
class TestTracing(unittest.TestCase):
    def testMaskRCNNFPN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    def testMaskRCNNFPN_with_postproc(self):
        def inference_func(model, image):
            inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]
            return model.inference(inputs, do_postprocess=True)[0]["instances"]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    @skipIfOnCPUCI
    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml", inference_func)

    @skipIfOnCPUCI
    def testCascadeRCNN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    # bug fixed by https://github.com/pytorch/pytorch/pull/67734
    @unittest.skipIf(TORCH_VERSION == (1, 10) and os.environ.get("CI"), "1.10 has bugs.")
    def testRetinaNet(self):
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func)

    def _check_torchscript_no_hardcoded_device(self, jitfile, extract_dir, device):
        zipfile.ZipFile(jitfile).extractall(extract_dir)
        dir_path = os.path.join(extract_dir, os.path.splitext(os.path.basename(jitfile))[0])
        error_files = []
        for f in glob.glob(f"{dir_path}/code/**/*.py", recursive=True):
            content = open(f).read()
            if device in content:
                error_files.append((f, content))
        if len(error_files):
            msg = "\n".join(f"{f}\n{content}" for f, content in error_files)
            raise ValueError(f"Found device '{device}' in following files:\n{msg}")

    def _get_device_casting_test_cases(self, model):
        # Indexing operation can causes hardcoded device type before 1.10
        if not TORCH_VERSION >= (1, 10) or torch.cuda.device_count() == 0:
            return [None]

        testing_devices = ["cpu", "cuda:0"]
        if torch.cuda.device_count() > 1:
            testing_devices.append(f"cuda:{torch.cuda.device_count() - 1}")
        assert str(model.device) in testing_devices
        testing_devices.remove(str(model.device))
        testing_devices = [None] + testing_devices  # test no casting first

        return testing_devices

    def _test_model(self, config_path, inference_func, batch=1):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))

        wrapper = TracingAdapter(model, inputs, inference_func)
        wrapper.eval()
        with torch.no_grad():
            # trace with smaller images, and the trace must still work
            trace_inputs = tuple(
                nn.functional.interpolate(image, scale_factor=random.uniform(0.5, 0.7))
                for _ in range(batch)
            )
            traced_model = torch.jit.trace(wrapper, trace_inputs)

        testing_devices = self._get_device_casting_test_cases(model)
        # save and load back the model in order to show traceback of TorchScript
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            basename = "model"
            jitfile = f"{d}/{basename}.jit"
            torch.jit.save(traced_model, jitfile)
            traced_model = torch.jit.load(jitfile)

            if any(device and "cuda" in device for device in testing_devices):
                self._check_torchscript_no_hardcoded_device(jitfile, d, "cuda")

        for device in testing_devices:
            print(f"Testing casting to {device} for inference (traced on {model.device}) ...")
            with torch.no_grad():
                outputs = inference_func(copy.deepcopy(model).to(device), *inputs)
                traced_outputs = wrapper.outputs_schema(traced_model.to(device)(*inputs))
            if batch > 1:
                for output, traced_output in zip(outputs, traced_outputs):
                    assert_instances_allclose(output, traced_output, size_as_tensor=True)
            else:
                assert_instances_allclose(outputs, traced_outputs, size_as_tensor=True)

    @skipIfOnCPUCI
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )

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
