# Copyright (c) Facebook, Inc. and its affiliates.

import io
import os
import unittest
import onnx
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import add_export_config
from detectron2.export.flatten import TracingAdapter
from detectron2.modeling import build_model
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import get_sample_coco_image

SLOW_PUBLIC_CPU_TEST = unittest.skipIf(
    os.environ.get("CI") and not torch.cuda.is_available(),
    "The test is too slow on CPUs and will be executed on CircleCI's GPU jobs.",
)

SUPPORTED_ONNX_OPSET = 14


# TODO: this test requires manifold access, see: T88318502
class TestONNXTracing(unittest.TestCase):
    def testMaskRCNNFPN(self):
        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    @SLOW_PUBLIC_CPU_TEST
    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml", inference_func
        )

    @SLOW_PUBLIC_CPU_TEST
    def testCascadeRCNN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model_zoo_from_config_path(
            "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    # # bug fixed by https://github.com/pytorch/pytorch/pull/67734
    @unittest.skipIf(TORCH_VERSION == (1, 10) and os.environ.get("CI"), "1.10 has bugs.")
    def testRetinaNet(self):
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        self._test_model_zoo_from_config_path(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func
        )

    def _test_model(self, model, inputs):
        f = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                f,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                opset_version=SUPPORTED_ONNX_OPSET,
            )
        assert onnx.load_from_string(f.getvalue())

    def _test_model_zoo_from_config_path(self, config_path, inference_func, batch=1):
        # TODO: Using device='cuda' raises
        #   RuntimeError: isBool() INTERNAL ASSERT FAILED at
        #   "/github/pytorch/torch/include/ATen/core/ivalue.h":590,
        #   please report a bug to PyTorch.
        #   from /github/pytorch/torch/_ops.py:142 (return of OpOverloadPacket.__call__)
        model = model_zoo.get(config_path, trained=True, device="cpu")
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        return self._test_model(adapter_model, adapter_model.flattened_inputs)

    def _test_model_from_config_path(self, config_path, inference_func, batch=1):
        from projects.PointRend import point_rend

        cfg = get_cfg()
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg = add_export_config(cfg)
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()

        model = build_model(cfg)

        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        return self._test_model(adapter_model, adapter_model.flattened_inputs)

    @SLOW_PUBLIC_CPU_TEST
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )
