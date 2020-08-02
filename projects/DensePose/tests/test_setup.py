# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

from .common import (
    get_config_files,
    get_evolution_config_files,
    get_hrnet_config_files,
    get_quick_schedules_config_files,
    setup,
)


class TestSetup(unittest.TestCase):
    def _test_setup(self, config_file):
        setup(config_file)

    def test_setup_configs(self):
        config_files = get_config_files()
        for config_file in config_files:
            self._test_setup(config_file)

    def test_setup_evolution_configs(self):
        config_files = get_evolution_config_files()
        for config_file in config_files:
            self._test_setup(config_file)

    def test_setup_hrnet_configs(self):
        config_files = get_hrnet_config_files()
        for config_file in config_files:
            self._test_setup(config_file)

    def test_setup_quick_schedules_configs(self):
        config_files = get_quick_schedules_config_files()
        for config_file in config_files:
            self._test_setup(config_file)
