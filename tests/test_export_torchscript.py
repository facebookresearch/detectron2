# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
import unittest
import torch
from torch import Tensor, nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.torchscript import dump_torchscript_IR, export_torchscript_with_instances
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import assert_instances_allclose, get_sample_coco_image, random_boxes


"""
https://detectron2.readthedocs.io/tutorials/deployment.html
contains some explanations of this file.
"""


@unittest.skipIf(os.environ.get("CI") or TORCH_VERSION < (1, 8), "Insufficient Pytorch version")
class TestScripting(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNN(self):
        self._test_rcnn_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
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
        script_model = export_torchscript_with_instances(model, fields)

        inputs = [{"image": get_sample_coco_image()}]
        with torch.no_grad():
            instance = model.inference(inputs, do_postprocess=False)[0]
            scripted_instance = script_model.inference(inputs, do_postprocess=False)[
                0
            ].to_instances()
        assert_instances_allclose(instance, scripted_instance)

    def _test_retinanet_model(self, config_path):
        model = model_zoo.get(config_path, trained=True)
        model.eval()

        fields = {
            "pred_boxes": Boxes,
            "scores": Tensor,
            "pred_classes": Tensor,
        }
        script_model = export_torchscript_with_instances(model, fields)

        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            instance = model(inputs)[0]["instances"]
            scripted_instance = script_model(inputs)[0].to_instances()
            scripted_instance = detector_postprocess(scripted_instance, img.shape[1], img.shape[2])
        assert_instances_allclose(instance, scripted_instance)
        # Note that the model currently cannot be saved and loaded into a new process:
        # https://github.com/pytorch/pytorch/issues/46944


@unittest.skipIf(os.environ.get("CI") or TORCH_VERSION < (1, 8), "Insufficient Pytorch version")
class TestTracing(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNN(self):
        class WrapModel(nn.ModuleList):
            def forward(self, image):
                inputs = [{"image": image}]
                outputs = self[0].inference(inputs, do_postprocess=False)[0]
                size = outputs.image_size
                if torch.jit.is_tracing():
                    assert isinstance(size, torch.Tensor)
                else:
                    size = torch.as_tensor(size)
                return (
                    size,
                    outputs.pred_classes,
                    outputs.pred_boxes.tensor,
                    outputs.scores,
                    outputs.pred_masks,
                )

            @staticmethod
            def convert_output(output):
                r = Instances(tuple(output[0]))
                r.pred_classes = output[1]
                r.pred_boxes = Boxes(output[2])
                r.scores = output[3]
                r.pred_masks = output[4]
                return r

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", WrapModel)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testRetinaNet(self):
        class WrapModel(nn.ModuleList):
            def forward(self, image):
                inputs = [{"image": image}]
                outputs = self[0].forward(inputs)[0]["instances"]
                size = outputs.image_size
                if torch.jit.is_tracing():
                    assert isinstance(size, torch.Tensor)
                else:
                    size = torch.as_tensor(size)
                return (
                    size,
                    outputs.pred_classes,
                    outputs.pred_boxes.tensor,
                    outputs.scores,
                )

            @staticmethod
            def convert_output(output):
                r = Instances(tuple(output[0]))
                r.pred_classes = output[1]
                r.pred_boxes = Boxes(output[2])
                r.scores = output[3]
                return r

        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml", WrapModel)

    def _test_model(self, config_path, WrapperCls):
        # TODO wrapper should be handled by export API in the future
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()

        model = WrapperCls([model])
        model.eval()
        with torch.no_grad(), patch_builtin_len():
            small_image = nn.functional.interpolate(image, scale_factor=0.5)
            # trace with a different image, and the trace must still work
            traced_model = torch.jit.trace(model, (small_image,))

            output = WrapperCls.convert_output(model(image))
            traced_output = WrapperCls.convert_output(traced_model(image))
        assert_instances_allclose(output, traced_output)

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
