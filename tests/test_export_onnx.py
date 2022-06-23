# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest
import warnings
import torch
from torch.hub import _check_module_exists

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import STABLE_ONNX_OPSET_VERSION
from detectron2.export.flatten import TracingAdapter
from detectron2.modeling import build_model
from detectron2.utils.testing import (
    _pytorch1111_symbolic_opset9_repeat_interleave,
    _pytorch1111_symbolic_opset9_to,
    get_sample_coco_image,
    register_custom_op_onnx_export,
    skipIfOnCPUCI,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedMinTorchVersion,
    unregister_custom_op_onnx_export,
)


@unittest.skipIf(not _check_module_exists("onnx"), "ONNX not installed.")
@skipIfUnsupportedMinTorchVersion("1.10")
class TestONNXTracingExport(unittest.TestCase):
    def testMaskRCNNFPN(self):
        def inference_func(model, images):
            with warnings.catch_warnings(record=True):
                inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    @skipIfOnCPUCI
    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml", inference_func
        )

    @skipIfOnCPUCI
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

    @skipIfOnCPUCI
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )

    @skipIfUnsupportedMinOpsetVersion(16, STABLE_ONNX_OPSET_VERSION)
    @skipIfUnsupportedMinTorchVersion("1.11.1")
    def testMaskRCNNFPN_with_postproc(self):
        def inference_func(model, image):
            inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]
            return model.inference(inputs, do_postprocess=True)[0]["instances"]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            inference_func,
            opset_version=STABLE_ONNX_OPSET_VERSION,
        )

    ################################################################################
    # Testcase internals - DO NOT add tests below this point
    ################################################################################

    def setUp(self):
        register_custom_op_onnx_export("::to", _pytorch1111_symbolic_opset9_to, 9, "1.11.1")
        register_custom_op_onnx_export(
            "::repeat_interleave",
            _pytorch1111_symbolic_opset9_repeat_interleave,
            9,
            "1.11.1",
        )

    def tearDown(self):
        unregister_custom_op_onnx_export("::to", 9, "1.11.1")
        unregister_custom_op_onnx_export("::repeat_interleave", 9, "1.11.1")

    def _test_model(
        self,
        model,
        inputs,
        inference_func=None,
        opset_version=STABLE_ONNX_OPSET_VERSION,
        save_onnx_graph_path=None,
        **export_kwargs,
    ):
        import onnx  # isort:skip

        f = io.BytesIO()
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        with torch.no_grad():
            try:
                torch.onnx.enable_log()
            except AttributeError:
                # Older ONNX versions does not have this API
                pass
            torch.onnx.export(
                adapter_model,
                adapter_model.flattened_inputs,
                f,
                training=torch.onnx.TrainingMode.EVAL,
                opset_version=opset_version,
                verbose=True,
                **export_kwargs,
            )
        onnx_model = onnx.load_from_string(f.getvalue())
        assert onnx_model is not None
        if save_onnx_graph_path:
            onnx.save(onnx_model, save_onnx_graph_path)

    def _test_model_zoo_from_config_path(
        self,
        config_path,
        inference_func,
        batch=1,
        opset_version=STABLE_ONNX_OPSET_VERSION,
        save_onnx_graph_path=None,
        **export_kwargs,
    ):
        model = model_zoo.get(config_path, trained=True)
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        return self._test_model(
            model, inputs, inference_func, opset_version, save_onnx_graph_path, **export_kwargs
        )

    def _test_model_from_config_path(
        self,
        config_path,
        inference_func,
        batch=1,
        opset_version=STABLE_ONNX_OPSET_VERSION,
        save_onnx_graph_path=None,
        **export_kwargs,
    ):
        from projects.PointRend import point_rend  # isort:skip

        cfg = get_cfg()
        cfg.DATALOADER.NUM_WORKERS = 0
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.freeze()

        model = build_model(cfg)

        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        return self._test_model(
            model, inputs, inference_func, opset_version, save_onnx_graph_path, **export_kwargs
        )
