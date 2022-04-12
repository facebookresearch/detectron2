# Copyright (c) Facebook, Inc. and its affiliates.

import io
import os
import unittest
import warnings
from functools import wraps
import onnx
import torch
from torch import nn
from torch._C import OptionalType
from torch.onnx import register_custom_op_symbolic, unregister_custom_op_symbolic
from torch.onnx.symbolic_registry import is_registered_op

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import add_export_config
from detectron2.export.flatten import TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import get_sample_coco_image, random_boxes

SLOW_PUBLIC_CPU_TEST = unittest.skipIf(
    os.environ.get("CI") and not torch.cuda.is_available(),
    "The test is too slow on CPUs and will be executed on CircleCI's GPU jobs.",
)

NOT_IMPLEMENTED_YET = unittest.skipIf(
    True,
    "Test is still in development and not stable for running.",
)

################################################################################
# Fake symbolics to allow exporting to succeed
################################################################################

SUPPORTED_ONNX_OPSET = 14


def stube_fake_symbolic(symbolic_name, opset_version, symbolic):
    def decorator(fn):
        fn.symbolic_name = symbolic_name
        fn.opset_version = opset_version

        @wraps(fn)
        def wrapper(g, *args, **kwargs):
            need_unregister = False
            ns, op_name = fn.symbolic_name.split("::")
            if not is_registered_op(op_name, ns, fn.opset_version):
                warnings.warn(f"Registering dummy `{fn.symbolic_name}-{fn.opset_version}` symbolic")
                need_unregister = True
                register_custom_op_symbolic(fn.symbolic_name, symbolic, fn.opset_version)
            ret = fn(g, *args, **kwargs)
            if need_unregister:
                unregister_custom_op_symbolic(fn.symbolic_name, fn.opset_version)
            return ret

        return wrapper

    return decorator


# def resolve_conj(g, self):
#     # using `_caffe2` domain to skip shape inference check
#     n = g.op("_caffe2::Constant", value_t=torch.tensor([.0]))
#     n.setType(OptionalType.ofTensor())
#     return self

# def resolve_neg(g, self):
#     # using `_caffe2` domain to skip shape inference check
#     n = g.op("_caffe2::Constant", value_t=torch.tensor([.0]))
#     n.setType(OptionalType.ofTensor())
#     return self


def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners):
    # using `_caffe2` domain to skip shape inference check
    n = g.op("_caffe2::Constant", value_t=torch.tensor([0.0]))
    n.setType(OptionalType.ofTensor())
    return n


################################################################################
# Tests
################################################################################


# TODO: this test requires manifold access, see: T88318502
class TestONNXTracing(unittest.TestCase):
    @stube_fake_symbolic("::grid_sampler", SUPPORTED_ONNX_OPSET, grid_sampler)
    def testPointRendRCNNFPN(self):
        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model_from_config_path(
            "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml",
            inference_func,
        )

    def testMaskRCNNFPN(self):
        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )

    # TODO: Need to add aten::grid_sampler, aten::resolve_conj and aten::resolve_neg symbolics
    @NOT_IMPLEMENTED_YET
    def testMaskRCNNFPN_with_postproc(self):
        def inference_func(model, image):
            inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]
            return model.inference(inputs, do_postprocess=True)[0]["instances"]

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

    # TODO: NotImplementedError: Unsupported aten::index operator of advanced
    #       indexing on tensor of unknown rank, try turning on shape and type
    #       propagate during export: torch.onnx._export(..., propagate=True).
    @NOT_IMPLEMENTED_YET
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

        # TODO: Try using TracingAdapter instead
        with patch_builtin_len():
            return self._test_model(model, gen_input(15, 15))
