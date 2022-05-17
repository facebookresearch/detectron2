# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest
import onnx
import torch
from torch.hub import _check_module_exists

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import SUPPORTED_ONNX_OPSET
from detectron2.export.flatten import TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.testing import get_sample_coco_image, random_boxes, skip_on_cpu_ci

from ._pytorch_monkey_patches import (
    _pytorch1111_symbolic_opset9_repeat_interleave,
    _pytorch1111_symbolic_opset9_to,
)
from .helper import (
    _register_custom_op_onnx_export,
    _unregister_custom_op_onnx_export,
    min_torch_version,
)


@unittest.skipIf(not _check_module_exists("onnx"), "ONNX not installed.")
@unittest.skipIf(not min_torch_version("1.10"), "PyTorch 1.10 is the minimum version")
class TestONNXTracingExport(unittest.TestCase):
    def testMaskRCNNFPN(self):
        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    @skip_on_cpu_ci
    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml", inference_func
        )

    @skip_on_cpu_ci
    @unittest.skipIf(not min_torch_version("1.10"), "PyTorch 1.10 is the minimum version")
    def testCascadeRCNN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model_zoo_from_config_path(
            "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    def testRetinaNet(self):
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        self._test_model_zoo_from_config_path(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func
        )

    @skip_on_cpu_ci
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )

    @unittest.skipIf(not min_torch_version("1.11.1"), "PyTorch 1.11.1+ is the minimum version")
    def testMaskRCNNFPN_with_postproc(self):
        def inference_func(model, image):
            inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]
            return model.inference(inputs, do_postprocess=True)[0]["instances"]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, opset_version=16
        )

    ################################################################################
    # Testcase internals - DO NOT add tests below this point
    ################################################################################

    def setUp(self):
        _register_custom_op_onnx_export(
            "::to", _pytorch1111_symbolic_opset9_to, SUPPORTED_ONNX_OPSET, "1.11.1"
        )
        _register_custom_op_onnx_export(
            "::repeat_interleave",
            _pytorch1111_symbolic_opset9_repeat_interleave,
            SUPPORTED_ONNX_OPSET,
            "1.11.1",
        )

    def tearDown(self):
        _unregister_custom_op_onnx_export("::to", SUPPORTED_ONNX_OPSET, "1.11.1")
        _unregister_custom_op_onnx_export("::repeat_interleave", SUPPORTED_ONNX_OPSET, "1.11.1")

    def _test_model(self, model, inputs, inference_func=None, opset_version=SUPPORTED_ONNX_OPSET):
        f = io.BytesIO()
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                adapter_model,
                adapter_model.flattened_inputs,
                f,
                training=torch.onnx.TrainingMode.EVAL,
                opset_version=opset_version,
            )
        assert onnx.load_from_string(f.getvalue())

    def _test_model_zoo_from_config_path(
        self, config_path, inference_func, batch=1, opset_version=SUPPORTED_ONNX_OPSET
    ):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        return self._test_model(model, inputs, inference_func, opset_version)

    def _test_model_from_config_path(
        self, config_path, inference_func, batch=1, opset_version=SUPPORTED_ONNX_OPSET
    ):
        from projects.PointRend import point_rend

        cfg = get_cfg()
        cfg.DATALOADER.NUM_WORKERS = 0
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.freeze()

        model = build_model(cfg)

        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        return self._test_model(model, inputs, inference_func, opset_version)
