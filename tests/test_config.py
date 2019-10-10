#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import tempfile
import unittest

from detectron2.config import downgrade_config, get_cfg, upgrade_config

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
