# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest

from detectron2 import model_zoo
from detectron2.modeling import FPN, GeneralizedRCNN

logger = logging.getLogger(__name__)


class TestModelZoo(unittest.TestCase):
    def test_get_returns_model(self):
        model = model_zoo.get("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", trained=False)
        assert isinstance(model, GeneralizedRCNN), model
        assert isinstance(model.backbone, FPN), model.backbone

    def test_get_invalid_model(self):
        self.assertRaises(RuntimeError, model_zoo.get, "Invalid/config.yaml")


if __name__ == "__main__":
    unittest.main()
