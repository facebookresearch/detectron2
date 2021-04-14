# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import unittest

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling import FPN, GeneralizedRCNN

logger = logging.getLogger(__name__)


class TestModelZoo(unittest.TestCase):
    def test_get_returns_model(self):
        model = model_zoo.get("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", trained=False)
        self.assertIsInstance(model, GeneralizedRCNN)
        self.assertIsInstance(model.backbone, FPN)

    def test_get_invalid_model(self):
        self.assertRaises(RuntimeError, model_zoo.get, "Invalid/config.yaml")

    def test_get_url(self):
        url = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
        self.assertEqual(
            url,
            "https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn/138602908/model_final_01ca85.pkl",  # noqa
        )
        url2 = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.py")
        self.assertEqual(url, url2)

    def _build_lazy_model(self, name):
        cfg = model_zoo.get_config("common/models/" + name)
        instantiate(cfg.model)

    def test_mask_rcnn_fpn(self):
        self._build_lazy_model("mask_rcnn_fpn.py")

    def test_mask_rcnn_c4(self):
        self._build_lazy_model("mask_rcnn_c4.py")

    def test_panoptic_fpn(self):
        self._build_lazy_model("panoptic_fpn.py")

    def test_schedule(self):
        cfg = model_zoo.get_config("common/coco_schedule.py")
        for _, v in cfg.items():
            instantiate(v)


if __name__ == "__main__":
    unittest.main()
