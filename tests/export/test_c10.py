# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

try:
    # Caffe2 used to be included in PyTorch, but since PyTorch 1.10+,
    # it is not included in pre-built packages. This is a safety BC check
    from detectron2.config import get_cfg
    from detectron2.export.c10 import Caffe2RPN
    from detectron2.layers import ShapeSpec
except ImportError:
    raise unittest.SkipTest(
        f"PyTorch does not have Caffe2 support. Skipping all tests in {__name__}"
    ) from None


class TestCaffe2RPN(unittest.TestCase):
    def test_instantiation(self):
        cfg = get_cfg()
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
        input_shapes = {"res4": ShapeSpec(channels=256, stride=4)}
        rpn = Caffe2RPN(cfg, input_shapes)
        assert rpn is not None
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
        with self.assertRaises(AssertionError):
            rpn = Caffe2RPN(cfg, input_shapes)
