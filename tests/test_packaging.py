# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

from detectron2.utils.collect_env import collect_env_info


class TestProjects(unittest.TestCase):
    def test_import(self):
        from detectron2.projects import point_rend

        _ = point_rend.add_pointrend_config

        import detectron2.projects.deeplab as deeplab

        _ = deeplab.add_deeplab_config


class TestCollectEnv(unittest.TestCase):
    def test(self):
        _ = collect_env_info()
