# Copyright (c) Facebook, Inc. and its affiliates.

import io
import os
import unittest
import torch
from torch.hub import _check_module_exists

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import SUPPORTED_ONNX_OPSET, add_export_config
from detectron2.export.flatten import TracingAdapter
from detectron2.modeling import build_model
from detectron2.utils.testing import (
    get_sample_coco_image,
    pytorch_112_symbolic_opset9_repeat_interleave,
    pytorch_112_symbolic_opset9_to,
    register_custom_op_onnx_export,
    skip_on_cpu_ci,
)


@unittest.skipIf(not _check_module_exists("onnx"), "ONNX not installed.")
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

    def _test_model(self, model, inputs):
        import onnx  # noqa: F401

        f = io.BytesIO()
        with torch.no_grad(), register_custom_op_onnx_export(
            "::to", pytorch_112_symbolic_opset9_to, SUPPORTED_ONNX_OPSET, "1.12"
        ), register_custom_op_onnx_export(
            "::repeat_interleave",
            pytorch_112_symbolic_opset9_repeat_interleave,
            SUPPORTED_ONNX_OPSET,
            "1.12",
        ):
            print(' lets export')
            torch.onnx.export(
                model,
                inputs,
                f,
                training=torch.onnx.TrainingMode.EVAL,
                opset_version=SUPPORTED_ONNX_OPSET,
            )
            print(' export went through')
        assert onnx.load_from_string(f.getvalue())

    def _test_model_zoo_from_config_path(self, config_path, inference_func, batch=1):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        return self._test_model(adapter_model, adapter_model.flattened_inputs)

    def _test_model_from_config_path(self, config_path, inference_func, batch=1):
        from projects.PointRend import point_rend

        cfg = get_cfg()
        cfg.DATALOADER.NUM_WORKERS = 0
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.freeze()

        model = build_model(cfg)

        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        return self._test_model(adapter_model, adapter_model.flattened_inputs)

    @skip_on_cpu_ci
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )
