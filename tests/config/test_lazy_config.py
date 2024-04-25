# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import tempfile
from itertools import count

from detectron2.config import LazyConfig, LazyCall as L
from omegaconf import DictConfig


class TestLazyPythonConfig(unittest.TestCase):
    def setUp(self):
        self.curr_dir = os.path.dirname(__file__)
        self.root_filename = os.path.join(self.curr_dir, "root_cfg.py")

    def test_load(self):
        cfg = LazyConfig.load(self.root_filename)

        self.assertEqual(cfg.dir1a_dict.a, "modified")
        self.assertEqual(cfg.dir1b_dict.a, 1)
        self.assertEqual(cfg.lazyobj.x, "base_a_1")

        cfg.lazyobj.x = "new_x"
        # reload
        cfg = LazyConfig.load(self.root_filename)
        self.assertEqual(cfg.lazyobj.x, "base_a_1")

    def test_save_load(self):
        cfg = LazyConfig.load(self.root_filename)
        with tempfile.TemporaryDirectory(prefix="detectron2") as d:
            fname = os.path.join(d, "test_config.yaml")
            LazyConfig.save(cfg, fname)
            cfg2 = LazyConfig.load(fname)

        self.assertEqual(cfg2.lazyobj._target_, "itertools.count")
        self.assertEqual(cfg.lazyobj._target_, count)
        cfg2.lazyobj.pop("_target_")
        cfg.lazyobj.pop("_target_")
        # the rest are equal
        self.assertEqual(cfg, cfg2)

    def test_failed_save(self):
        cfg = DictConfig({"x": lambda: 3}, flags={"allow_objects": True})
        with tempfile.TemporaryDirectory(prefix="detectron2") as d:
            fname = os.path.join(d, "test_config.yaml")
            LazyConfig.save(cfg, fname)
            self.assertTrue(os.path.exists(fname))
            self.assertTrue(os.path.exists(fname + ".pkl"))

    def test_overrides(self):
        cfg = LazyConfig.load(self.root_filename)
        LazyConfig.apply_overrides(cfg, ["lazyobj.x=123", 'dir1b_dict.a="123"'])
        self.assertEqual(cfg.dir1b_dict.a, "123")
        self.assertEqual(cfg.lazyobj.x, 123)

        LazyConfig.apply_overrides(cfg, ["dir1b_dict.a='abc'"])
        self.assertEqual(cfg.dir1b_dict.a, "abc")

    def test_invalid_overrides(self):
        cfg = LazyConfig.load(self.root_filename)
        with self.assertRaises(KeyError):
            LazyConfig.apply_overrides(cfg, ["lazyobj.x.xxx=123"])

    def test_to_py(self):
        cfg = LazyConfig.load(self.root_filename)
        cfg.lazyobj.x = {"a": 1, "b": 2, "c": L(count)(x={"r": "a", "s": 2.4, "t": [1, 2, 3, "z"]})}
        cfg.list = ["a", 1, "b", 3.2]
        py_str = LazyConfig.to_py(cfg)
        expected = """cfg.dir1a_dict.a = "modified"
cfg.dir1a_dict.b = 2
cfg.dir1b_dict.a = 1
cfg.dir1b_dict.b = 2
cfg.lazyobj = itertools.count(
    x={
        "a": 1,
        "b": 2,
        "c": itertools.count(x={"r": "a", "s": 2.4, "t": [1, 2, 3, "z"]}),
    },
    y="base_a_1_from_b",
)
cfg.list = ["a", 1, "b", 3.2]
"""
        self.assertEqual(py_str, expected)

    def test_bad_import(self):
        file = os.path.join(self.curr_dir, "dir1", "bad_import.py")
        with self.assertRaisesRegex(ImportError, "relative import"):
            LazyConfig.load(file)

    def test_bad_import2(self):
        file = os.path.join(self.curr_dir, "dir1", "bad_import2.py")
        with self.assertRaisesRegex(ImportError, "not exist"):
            LazyConfig.load(file)

    def test_load_rel(self):
        file = os.path.join(self.curr_dir, "dir1", "load_rel.py")
        cfg = LazyConfig.load(file)
        self.assertIn("x", cfg)
