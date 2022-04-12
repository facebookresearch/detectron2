# Copyright (c) Facebook, Inc. and its affiliates.

import io
import onnx
import os
import unittest
import torch
import warnings
from torch import nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.flatten import TracingAdapter
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import (
    get_sample_coco_image,
    random_boxes,
)


SLOW_PUBLIC_CPU_TEST = unittest.skipIf(
    os.environ.get("CI") and not torch.cuda.is_available(),
    "The test is too slow on CPUs and will be executed on CircleCI's GPU jobs.",
)

SUPPORTED_ONNX_OPSET = 14
# try:
#     from torch.onnx.symbolic_registry import is_registered_op
#     from torch.onnx import register_custom_op_symbolic
#     from torch._C import OptionalType

#     if not is_registered_op("grid_sampler", "", SUPPORTED_ONNX_OPSET):
#         warnings.warn("Registering dummy `::grid_sampler` symbolic for testing purposes")
#         # Dummy symbolic
#         def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners):
#             n = g.op("_caffe2::Constant", value_t=torch.tensor([.0]))  # using _caffe2 domain to skip shape inference check
#             n.setType(OptionalType.ofTensor())
#             return n
#         # TODO: Unregister it later?
#     if not is_registered_op("resolve_conj", "", SUPPORTED_ONNX_OPSET):
#         warnings.warn("Registering dummy `::resolve_conj` symbolic for testing purposes")
#         # Dummy symbolic
#         def resolve_conj(g, self):
#             n = g.op("_caffe2::Constant", value_v=self)  # using _caffe2 domain to skip shape inference check
#             n.setType(OptionalType.ofTensor())
#             return self
#     if not is_registered_op("resolve_neg", "", SUPPORTED_ONNX_OPSET):
#         warnings.warn("Registering dummy `::resolve_neg` symbolic for testing purposes")
#         # Dummy symbolic
#         def resolve_neg(g, self):
#             n = g.op("_caffe2::Constant", value_v=self)  # using _caffe2 domain to skip shape inference check
#             n.setType(OptionalType.ofTensor())
#             return self
#         # TODO: Unregister it later?
#         register_custom_op_symbolic("::grid_sampler", grid_sampler, SUPPORTED_ONNX_OPSET)
#         register_custom_op_symbolic("::resolve_conj", resolve_conj, SUPPORTED_ONNX_OPSET)
#         register_custom_op_symbolic("::resolve_neg", resolve_neg, SUPPORTED_ONNX_OPSET)
# except RuntimeError:
#     print('@'*10000)
#     pass


# TODO: this test requires manifold access, see: T88318502
class TestONNXTracing(unittest.TestCase):
    def testMaskRCNNFPN(self):
        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    # TODO: Need to add aten::grid_sampler, aten::resolve_conj and aten::resolve_neg symbolics
    # def testMaskRCNNFPN_with_postproc(self):
    #     def inference_func(model, image):
    #         inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]
    #         out =  model.inference(inputs, do_postprocess=True)[0]["instances"]
    #         print(out)
    #         return out
    #     self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    # @SLOW_PUBLIC_CPU_TEST
    def testMaskRCNNC4(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml", inference_func)

    # @SLOW_PUBLIC_CPU_TEST
    def testCascadeRCNN(self):
        def inference_func(model, image):
            inputs = [{"image": image}]
            return model.inference(inputs, do_postprocess=False)[0]

        self._test_model("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", inference_func)

    # # bug fixed by https://github.com/pytorch/pytorch/pull/67734
    @unittest.skipIf(TORCH_VERSION == (1, 10) and os.environ.get("CI"), "1.10 has bugs.")
    def testRetinaNet(self):
        def inference_func(model, image):
            return model.forward([{"image": image}])[0]["instances"]

        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml", inference_func)

    def _test_model(self, config_path, inference_func, batch=1):
        # TODO: Using device='cuda' raises
        #   RuntimeError: isBool() INTERNAL ASSERT FAILED at "/github/pytorch/torch/include/ATen/core/ivalue.h":590, please report a bug to PyTorch.
        #   from /github/pytorch/torch/_ops.py:142 (return of OpOverloadPacket.__call__)
        if isinstance(config_path, str):
            model = model_zoo.get(config_path, trained=True, device='cpu')
        else:
            model = config_path
        image = get_sample_coco_image()
        inputs = tuple(image.clone() for _ in range(batch))
        adapter_model = TracingAdapter(model, inputs, inference_func)
        adapter_model.eval()
        f = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(adapter_model,
                              adapter_model.flattened_inputs,
                              f,
                              training=torch.onnx.TrainingMode.EVAL,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                              opset_version=SUPPORTED_ONNX_OPSET,
                              )
        loaded_model = onnx.load_from_string(f.getvalue())
        assert loaded_model

    # @SLOW_PUBLIC_CPU_TEST
    def testMaskRCNNFPN_batched(self):
        def inference_func(model, image1, image2):
            inputs = [{"image": image1}, {"image": image2}]
            return model.inference(inputs, do_postprocess=False)

        self._test_model(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func, batch=2
        )

    # TODO: NotImplementedError: Unsupported aten::index operator of advanced indexing on tensor of unknown rank,
    #       try turning on shape and type propagate during export: torch.onnx._export(..., propagate=True).
    # def testKeypointHead(self):
    #     class M(nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.model = KRCNNConvDeconvUpsampleHead(
    #                 ShapeSpec(channels=4, height=14, width=14), num_keypoints=17, conv_dims=(4,)
    #             )

    #         def forward(self, x, predbox1, predbox2):
    #             inst = [
    #                 Instances((100, 100), pred_boxes=Boxes(predbox1)),
    #                 Instances((100, 100), pred_boxes=Boxes(predbox2)),
    #             ]
    #             ret = self.model(x, inst)
    #             return tuple(x.pred_keypoints for x in ret)

    #     model = M()
    #     model.eval()

    #     def gen_input(num1, num2):
    #         feat = torch.randn((num1 + num2, 4, 14, 14))
    #         box1 = random_boxes(num1)
    #         box2 = random_boxes(num2)
    #         return feat, box1, box2

    #     adapter_model = TracingAdapter(model, gen_input(15, 15))
    #     adapter_model.eval()
    #     f = io.BytesIO()
    #     with torch.no_grad(), patch_builtin_len():
    #         torch.onnx.export(model,  # adapter_model,
    #                           gen_input(15, 15),  # adapter_model.flattened_inputs,
    #                           f,
    #                           training=torch.onnx.TrainingMode.EVAL,
    #                           operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    #                           opset_version=SUPPORTED_ONNX_OPSET,
    #                           verbose=True
    #                           )
    #     loaded_model = onnx.load_from_string(f.getvalue())
    #     assert loaded_model
