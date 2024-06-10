# Copyright (c) Facebook, Inc. and its affiliates.

import io
import unittest
import warnings
import onnx
import torch
from packaging import version
from torch.hub import _check_module_exists

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import STABLE_ONNX_OPSET_VERSION
from detectron2.export.flatten import TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.testing import (
    _pytorch1111_symbolic_opset9_repeat_interleave,
    _pytorch1111_symbolic_opset9_to,
    get_sample_coco_image,
    has_dynamic_axes,
    random_boxes,
    register_custom_op_onnx_export,
    skipIfOnCPUCI,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedMinTorchVersion,
    unregister_custom_op_onnx_export,
)


@unittest.skipIf(not _check_module_exists("onnx"), "ONNX not installed.")
@skipIfUnsupportedMinTorchVersion("1.10")
class TestONNXTracingExport(unittest.TestCase):
    opset_version = STABLE_ONNX_OPSET_VERSION

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
        )

    @unittest.skipIf(
        version.Version(onnx.version.version) >= version.Version("1.16.0"),
        "This test fails on ONNX Runtime >= 1.16",
    )
    def testKeypointHead(self):
        class M(torch.nn.Module):
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

        with patch_builtin_len():
            onnx_model = self._test_model(
                model,
                gen_input(1, 2),
                input_names=["features", "pred_boxes", "pred_classes"],
                output_names=["box1", "box2"],
                dynamic_axes={
                    "features": {0: "batch", 1: "static_four", 2: "height", 3: "width"},
                    "pred_boxes": {0: "batch", 1: "static_four"},
                    "pred_classes": {0: "batch", 1: "static_four"},
                    "box1": {0: "num_instance", 1: "K", 2: "static_three"},
                    "box2": {0: "num_instance", 1: "K", 2: "static_three"},
                },
            )

            # Although ONNX models are not executable by PyTorch to verify
            # support of batches with different sizes, we can verify model's IR
            # does not hard-code input and/or output shapes.
            # TODO: Add tests with different batch sizes when detectron2's CI
            #       support ONNX Runtime backend.
            assert has_dynamic_axes(onnx_model)

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
        # Not imported in the beginning of file to prevent runtime errors
        # for environments without ONNX.
        # This testcase checks dependencies before running
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
        return onnx_model

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
