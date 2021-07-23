#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.


import os
import tempfile
import unittest
import torch
from omegaconf import OmegaConf

from detectron2 import model_zoo
from detectron2.config import configurable, downgrade_config, get_cfg, upgrade_config
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model

_V0_CFG = """
MODEL:
  RPN_HEAD:
    NAME: "TEST"
VERSION: 0
"""

_V1_CFG = """
MODEL:
  WEIGHT: "/path/to/weight"
"""


class TestConfigVersioning(unittest.TestCase):
    def test_upgrade_downgrade_consistency(self):
        cfg = get_cfg()
        # check that custom is preserved
        cfg.USER_CUSTOM = 1

        down = downgrade_config(cfg, to_version=0)
        up = upgrade_config(down)
        self.assertTrue(up == cfg)

    def _merge_cfg_str(self, cfg, merge_str):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            f.write(merge_str)
            f.close()
            cfg.merge_from_file(f.name)
        finally:
            os.remove(f.name)
        return cfg

    def test_auto_upgrade(self):
        cfg = get_cfg()
        latest_ver = cfg.VERSION
        cfg.USER_CUSTOM = 1

        self._merge_cfg_str(cfg, _V0_CFG)

        self.assertEqual(cfg.MODEL.RPN.HEAD_NAME, "TEST")
        self.assertEqual(cfg.VERSION, latest_ver)

    def test_guess_v1(self):
        cfg = get_cfg()
        latest_ver = cfg.VERSION
        self._merge_cfg_str(cfg, _V1_CFG)
        self.assertEqual(cfg.VERSION, latest_ver)


class _TestClassA(torch.nn.Module):
    @configurable
    def __init__(self, arg1, arg2, arg3=3):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        assert arg1 == 1
        assert arg2 == 2
        assert arg3 == 3

    @classmethod
    def from_config(cls, cfg):
        args = {"arg1": cfg.ARG1, "arg2": cfg.ARG2}
        return args


class _TestClassB(_TestClassA):
    @configurable
    def __init__(self, input_shape, arg1, arg2, arg3=3):
        """
        Doc of _TestClassB
        """
        assert input_shape == "shape"
        super().__init__(arg1, arg2, arg3)

    @classmethod
    def from_config(cls, cfg, input_shape):  # test extra positional arg in from_config
        args = {"arg1": cfg.ARG1, "arg2": cfg.ARG2}
        args["input_shape"] = input_shape
        return args


class _LegacySubClass(_TestClassB):
    # an old subclass written in cfg style
    def __init__(self, cfg, input_shape, arg4=4):
        super().__init__(cfg, input_shape)
        assert self.arg1 == 1
        assert self.arg2 == 2
        assert self.arg3 == 3


class _NewSubClassNewInit(_TestClassB):
    # test new subclass with a new __init__
    @configurable
    def __init__(self, input_shape, arg4=4, **kwargs):
        super().__init__(input_shape, **kwargs)
        assert self.arg1 == 1
        assert self.arg2 == 2
        assert self.arg3 == 3


class _LegacySubClassNotCfg(_TestClassB):
    # an old subclass written in cfg style, but argument is not called "cfg"
    def __init__(self, config, input_shape):
        super().__init__(config, input_shape)
        assert self.arg1 == 1
        assert self.arg2 == 2
        assert self.arg3 == 3


class _TestClassC(_TestClassB):
    @classmethod
    def from_config(cls, cfg, input_shape, **kwargs):  # test extra kwarg overwrite
        args = {"arg1": cfg.ARG1, "arg2": cfg.ARG2}
        args["input_shape"] = input_shape
        args.update(kwargs)
        return args


class _TestClassD(_TestClassA):
    @configurable
    def __init__(self, input_shape: ShapeSpec, arg1: int, arg2, arg3=3):
        assert input_shape == "shape"
        super().__init__(arg1, arg2, arg3)

    # _TestClassA.from_config does not have input_shape args.
    # Test whether input_shape will be forwarded to __init__


@configurable(from_config=lambda cfg, arg2: {"arg1": cfg.ARG1, "arg2": arg2, "arg3": cfg.ARG3})
def _test_func(arg1, arg2=2, arg3=3, arg4=4):
    return arg1, arg2, arg3, arg4


class TestConfigurable(unittest.TestCase):
    def testInitWithArgs(self):
        _ = _TestClassA(arg1=1, arg2=2, arg3=3)
        _ = _TestClassB("shape", arg1=1, arg2=2)
        _ = _TestClassC("shape", arg1=1, arg2=2)
        _ = _TestClassD("shape", arg1=1, arg2=2, arg3=3)

    def testPatchedAttr(self):
        self.assertTrue("Doc" in _TestClassB.__init__.__doc__)
        self.assertEqual(_TestClassD.__init__.__annotations__["arg1"], int)

    def testInitWithCfg(self):
        cfg = get_cfg()
        cfg.ARG1 = 1
        cfg.ARG2 = 2
        cfg.ARG3 = 3
        _ = _TestClassA(cfg)
        _ = _TestClassB(cfg, input_shape="shape")
        _ = _TestClassC(cfg, input_shape="shape")
        _ = _TestClassD(cfg, input_shape="shape")
        _ = _LegacySubClass(cfg, input_shape="shape")
        _ = _NewSubClassNewInit(cfg, input_shape="shape")
        _ = _LegacySubClassNotCfg(cfg, input_shape="shape")
        with self.assertRaises(TypeError):
            # disallow forwarding positional args to __init__ since it's prone to errors
            _ = _TestClassD(cfg, "shape")

        # call with kwargs instead
        _ = _TestClassA(cfg=cfg)
        _ = _TestClassB(cfg=cfg, input_shape="shape")
        _ = _TestClassC(cfg=cfg, input_shape="shape")
        _ = _TestClassD(cfg=cfg, input_shape="shape")
        _ = _LegacySubClass(cfg=cfg, input_shape="shape")
        _ = _NewSubClassNewInit(cfg=cfg, input_shape="shape")
        _ = _LegacySubClassNotCfg(config=cfg, input_shape="shape")

    def testInitWithCfgOverwrite(self):
        cfg = get_cfg()
        cfg.ARG1 = 1
        cfg.ARG2 = 999  # wrong config
        with self.assertRaises(AssertionError):
            _ = _TestClassA(cfg, arg3=3)

        # overwrite arg2 with correct config later:
        _ = _TestClassA(cfg, arg2=2, arg3=3)
        _ = _TestClassB(cfg, input_shape="shape", arg2=2, arg3=3)
        _ = _TestClassC(cfg, input_shape="shape", arg2=2, arg3=3)
        _ = _TestClassD(cfg, input_shape="shape", arg2=2, arg3=3)

        # call with kwargs cfg=cfg instead
        _ = _TestClassA(cfg=cfg, arg2=2, arg3=3)
        _ = _TestClassB(cfg=cfg, input_shape="shape", arg2=2, arg3=3)
        _ = _TestClassC(cfg=cfg, input_shape="shape", arg2=2, arg3=3)
        _ = _TestClassD(cfg=cfg, input_shape="shape", arg2=2, arg3=3)

    def testInitWithCfgWrongArgs(self):
        cfg = get_cfg()
        cfg.ARG1 = 1
        cfg.ARG2 = 2
        with self.assertRaises(TypeError):
            _ = _TestClassB(cfg, "shape", not_exist=1)
        with self.assertRaises(TypeError):
            _ = _TestClassC(cfg, "shape", not_exist=1)
        with self.assertRaises(TypeError):
            _ = _TestClassD(cfg, "shape", not_exist=1)

    def testBadClass(self):
        class _BadClass1:
            @configurable
            def __init__(self, a=1, b=2):
                pass

        class _BadClass2:
            @configurable
            def __init__(self, a=1, b=2):
                pass

            def from_config(self, cfg):  # noqa
                pass

        class _BadClass3:
            @configurable
            def __init__(self, a=1, b=2):
                pass

            # bad name: must be cfg
            @classmethod
            def from_config(cls, config):  # noqa
                pass

        with self.assertRaises(AttributeError):
            _ = _BadClass1(a=1)

        with self.assertRaises(TypeError):
            _ = _BadClass2(a=1)

        with self.assertRaises(TypeError):
            _ = _BadClass3(get_cfg())

    def testFuncWithCfg(self):
        cfg = get_cfg()
        cfg.ARG1 = 10
        cfg.ARG3 = 30

        self.assertEqual(_test_func(1), (1, 2, 3, 4))
        with self.assertRaises(TypeError):
            _test_func(cfg)
        self.assertEqual(_test_func(cfg, arg2=2), (10, 2, 30, 4))
        self.assertEqual(_test_func(cfg, arg1=100, arg2=20), (100, 20, 30, 4))
        self.assertEqual(_test_func(cfg, arg1=100, arg2=20, arg4=40), (100, 20, 30, 40))

        self.assertTrue(callable(_test_func.from_config))

    def testOmegaConf(self):
        cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        cfg = OmegaConf.create(cfg.dump())
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        # test that a model can be built with omegaconf config as well
        build_model(cfg)
