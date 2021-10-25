# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import copy
import os
import tempfile
import unittest
import torch

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.testing import get_sample_coco_image


@unittest.skipIf(os.environ.get("CI"), "Require COCO data and model zoo.")
class TestCaffe2Export(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _test_model(self, config_path, device="cpu"):
        # requires extra dependencies
        from detectron2.export import Caffe2Model, add_export_config, Caffe2Tracer

        cfg = model_zoo.get_config(config_path)
        add_export_config(cfg)
        cfg.MODEL.DEVICE = device
        model = model_zoo.get(config_path, trained=True, device=device)

        inputs = [{"image": get_sample_coco_image()}]
        tracer = Caffe2Tracer(cfg, model, copy.deepcopy(inputs))

        c2_model = tracer.export_caffe2()

        with tempfile.TemporaryDirectory(prefix="detectron2_unittest") as d:
            c2_model.save_protobuf(d)
            c2_model.save_graph(os.path.join(d, "test.svg"), inputs=copy.deepcopy(inputs))

            c2_model = Caffe2Model.load_protobuf(d)
            c2_model(inputs)[0]["instances"]

            ts_model = tracer.export_torchscript()
            ts_model.save(os.path.join(d, "model.ts"))

    def testMaskRCNN(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNNGPU(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device="cuda")

    def testRetinaNet(self):
        # TODO: this test requires manifold access, see: T88318502
        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
