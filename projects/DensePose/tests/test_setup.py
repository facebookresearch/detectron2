# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import unittest

from detectron2.config import get_cfg
from detectron2.engine import default_setup

from densepose import add_densepose_config

_CONFIG_DIR = "configs"
_QUICK_SCHEDULES_CONFIG_SUB_DIR = "quick_schedules"
_CONFIG_FILE_PREFIX = "densepose_"
_CONFIG_FILE_EXT = ".yaml"


def _get_config_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", _CONFIG_DIR)


def _collect_config_files(config_dir):
    paths = []
    for entry in os.listdir(config_dir):
        _, ext = os.path.splitext(entry)
        if ext != _CONFIG_FILE_EXT:
            continue
        if not entry.startswith(_CONFIG_FILE_PREFIX):
            continue
        path = os.path.join(config_dir, entry)
        paths.append(path)
    return paths


def _get_config_files():
    config_dir = _get_config_dir()
    return _collect_config_files(config_dir)


def _get_quick_schedules_config_files():
    config_dir = _get_config_dir()
    config_dir = os.path.join(config_dir, _QUICK_SCHEDULES_CONFIG_SUB_DIR)
    return _collect_config_files(config_dir)


class TestSetup(unittest.TestCase):
    def _test_setup(self, config_file):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        default_setup(cfg, {})

    def test_setup_configs(self):
        config_files = _get_config_files()
        for config_file in config_files:
            self._test_setup(config_file)

    def test_setup_quick_schedules_configs(self):
        config_files = _get_quick_schedules_config_files()
        for config_file in config_files:
            self._test_setup(config_file)
