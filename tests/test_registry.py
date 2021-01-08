# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import torch

from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.registry import _convert_target_to_string, locate


class A:
    class B:
        pass


class TestLocate(unittest.TestCase):
    def _test_obj(self, obj):
        name = _convert_target_to_string(obj)
        newobj = locate(name)
        self.assertIs(obj, newobj)

    def test_basic(self):
        self._test_obj(GeneralizedRCNN)

    def test_inside_class(self):
        # requires using __qualname__ instead of __name__
        self._test_obj(A.B)

    def test_builtin(self):
        self._test_obj(len)
        self._test_obj(dict)

    def test_pytorch_optim(self):
        # pydoc.locate does not work for it
        self._test_obj(torch.optim.SGD)

    def test_failure(self):
        with self.assertRaises(ImportError):
            locate("asdf")
