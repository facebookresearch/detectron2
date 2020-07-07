# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-

import copy
import numpy as np
import os
import tempfile
import unittest
import cv2
import torch
from fvcore.common.file_io import PathManager

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger


@unittest.skipIf(os.environ.get("CIRCLECI"), "Require COCO data and model zoo.")
class TestCaffe2Export(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def _test_model(self, config_path, device="cpu"):
        # requires extra dependencies
        from detectron2.export import Caffe2Model, add_export_config, export_caffe2_model

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg = add_export_config(cfg)
        cfg.MODEL.DEVICE = device

        inputs = [{"image": self._get_test_image()}]
        model = build_model(cfg)
        DetectionCheckpointer(model).load(model_zoo.get_checkpoint_url(config_path))
        c2_model = export_caffe2_model(cfg, model, copy.deepcopy(inputs))

        with tempfile.TemporaryDirectory(prefix="detectron2_unittest") as d:
            c2_model.save_protobuf(d)
            c2_model.save_graph(os.path.join(d, "test.svg"), inputs=copy.deepcopy(inputs))
            c2_model = Caffe2Model.load_protobuf(d)
        c2_model(inputs)[0]["instances"]

    def _get_test_image(self):
        try:
            file_name = DatasetCatalog.get("coco_2017_train")[0]["file_name"]
            assert PathManager.exists(file_name)
        except Exception:
            self.skipTest("COCO dataset not available.")

        with PathManager.open(file_name, "rb") as f:
            buf = f.read()
        img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None, file_name
        return torch.from_numpy(img.transpose(2, 0, 1))

    def testMaskRCNN(self):
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def testMaskRCNNGPU(self):
        self._test_model("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", device="cuda")

    def testRetinaNet(self):
        self._test_model("COCO-Detection/retinanet_R_50_FPN_3x.yaml")

    def testPanopticFPN(self):
        self._test_model("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
