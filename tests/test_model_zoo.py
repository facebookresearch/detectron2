# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
from unittest import mock

from detectron2 import model_zoo
from detectron2.modeling import GeneralizedRCNN, ResNet

logger = logging.getLogger(__name__)


class TestModelZoo(unittest.TestCase):

    # Patch the URL mapping to only contain one config path.
    @mock.patch.object(
        model_zoo.ModelZooUrls,
        "CONFIG_PATH_TO_URL_SUFFIX", {
            "quick_schedules/mask_rcnn_R_50_C4_model_zoo_test.yaml": ""
        }
    )
    def test_get_returns_model(self):
        model = model_zoo.get(
            "quick_schedules/mask_rcnn_R_50_C4_model_zoo_test.yaml", trained=False
        )
        assert isinstance(model, GeneralizedRCNN)
        assert isinstance(model.backbone, ResNet)

    def test_get_invalid_model(self):
        self.assertRaises(RuntimeError, model_zoo.get, "Invalid/config.yaml")


if __name__ == "__main__":
    unittest.main()
