# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
import unittest
import yaml
from omegaconf import OmegaConf
from omegaconf import __version__ as oc_version
from dataclasses import dataclass

from detectron2.config import LazyConfig, instantiate, LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.utils.testing import reload_lazy_config

OC_VERSION = tuple(int(x) for x in oc_version.split(".")[:2])


class TestClass:
    def __init__(self, int_arg, list_arg=None, dict_arg=None, extra_arg=None):
        self.int_arg = int_arg
        self.list_arg = list_arg
        self.dict_arg = dict_arg
        self.extra_arg = extra_arg

    def __call__(self, call_arg):
        return call_arg + self.int_arg


@unittest.skipIf(OC_VERSION < (2, 1), "omegaconf version too old")
class TestConstruction(unittest.TestCase):
    def test_basic_construct(self):
        cfg = L(TestClass)(
            int_arg=3,
            list_arg=[10],
            dict_arg={},
            extra_arg=L(TestClass)(int_arg=4, list_arg="${..list_arg}"),
        )

        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            self.assertIsInstance(obj, TestClass)
            self.assertEqual(obj.int_arg, 3)
            self.assertEqual(obj.extra_arg.int_arg, 4)
            self.assertEqual(obj.extra_arg.list_arg, obj.list_arg)

            # Test interpolation
            x.extra_arg.list_arg = [5]
            obj = instantiate(x)
            self.assertIsInstance(obj, TestClass)
            self.assertEqual(obj.extra_arg.list_arg, [5])

    def test_instantiate_other_obj(self):
        # do nothing for other obj
        self.assertEqual(instantiate(5), 5)
        x = [3, 4, 5]
        self.assertEqual(instantiate(x), x)
        x = TestClass(1)
        self.assertIs(instantiate(x), x)
        x = {"xx": "yy"}
        self.assertIs(instantiate(x), x)

    def test_instantiate_lazy_target(self):
        # _target_ is result of instantiate
        objconf = L(L(len)(int_arg=3))(call_arg=4)
        objconf._target_._target_ = TestClass
        self.assertEqual(instantiate(objconf), 7)

    def test_instantiate_list(self):
        lst = [1, 2, L(TestClass)(int_arg=1)]
        x = L(TestClass)(int_arg=lst)  # list as an argument should be recursively instantiated
        x = instantiate(x).int_arg
        self.assertEqual(x[:2], [1, 2])
        self.assertIsInstance(x[2], TestClass)
        self.assertEqual(x[2].int_arg, 1)

    def test_instantiate_dataclass(self):
        cfg = L(ShapeSpec)(channels=1, width=3)
        # Test original cfg as well as serialization
        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            self.assertIsInstance(obj, ShapeSpec)
            self.assertEqual(obj.channels, 1)
            self.assertEqual(obj.height, None)

    def test_instantiate_dataclass_as_subconfig(self):
        cfg = L(TestClass)(int_arg=1, extra_arg=ShapeSpec(channels=1, width=3))
        # Test original cfg as well as serialization
        for x in [cfg, reload_lazy_config(cfg)]:
            obj = instantiate(x)
            self.assertIsInstance(obj.extra_arg, ShapeSpec)
            self.assertEqual(obj.extra_arg.channels, 1)
            self.assertEqual(obj.extra_arg.height, None)

    def test_bad_lazycall(self):
        with self.assertRaises(Exception):
            L(3)

    def test_interpolation(self):
        cfg = L(TestClass)(int_arg=3, extra_arg="${int_arg}")

        cfg.int_arg = 4
        obj = instantiate(cfg)
        self.assertEqual(obj.extra_arg, 4)

        # Test that interpolation still works after serialization
        cfg = reload_lazy_config(cfg)
        cfg.int_arg = 5
        obj = instantiate(cfg)
        self.assertEqual(obj.extra_arg, 5)
