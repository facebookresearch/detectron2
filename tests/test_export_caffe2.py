# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import copy
import os
import tempfile
import unittest
import torch
from torch.hub import _check_module_exists

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.testing import get_sample_coco_image

try:
    # Caffe2 used to be included in PyTorch, but since PyTorch 1.10+,
    # Caffe2 is not included in pre-built packages. This is a safety BC check
    from detectron2.export import Caffe2Model, Caffe2Tracer
except ImportError:
    raise unittest.SkipTest(
        f"PyTorch does not have Caffe2 support. Skipping all tests in {__name__}"
    ) from None


# TODO: this test requires manifold access, see: T88318502
# Running it on CircleCI causes crash, not sure why.
@unittest.skipIf(os.environ.get("CIRCLECI"), "Caffe2 tests crash on CircleCI.")
@unittest.skipIf(not _check_module_exists("onnx"), "ONNX not installed.")
class TestCaffe2Export(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _test_model(self, config_path, device="cpu"):
        cfg = model_zoo.get_config(config_path)
        cfg.MODEL.DEVICE = device
        model = model_zoo.get(config_path, trained=True, device=device)

        inputs = [{"image": get_sample_coco_image()}]
        tracer = Caffe2Tracer(cfg, model, copy.deepcopy(inputs))

        with tempfile.TemporaryDirectory(prefix="detectron2_unittest") as d:
            if not os.environ.get("CI"):
                # This requires onnx, which is not yet available on public CI
                c2_model = tracer.export_caffe2()
                c2_model.save_protobuf(d)
                c2_model.save_graph(os.path.join(d, "test.svg"), inputs=copy.deepcopy(inputs))

                c2_model = Caffe2Model.load_protobuf(d)
                c2_model(inputs)[0]["instances"]

            ts_model = tracer.export_torchscript()
            ts_model.save(os.path.join(d, "model.ts"))

    def testMaskRCNN(self):
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNNGPU(self):
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device="cuda")

    def testRetinaNet(self):
        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
