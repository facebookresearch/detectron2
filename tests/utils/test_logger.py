# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import tempfile
import unittest
import uuid

from detectron2.utils.logger import setup_logger


class TestSetupLogger(unittest.TestCase):
    def test_setup_logger_bare_filename(self):
        """setup_logger(output="train.log") must not crash on an empty dirname."""
        with tempfile.TemporaryDirectory(prefix="detectron2_tests") as tmp_dir:
            cwd = os.getcwd()
            name = "detectron2_tests_logger_" + uuid.uuid4().hex
            try:
                os.chdir(tmp_dir)
                logger = setup_logger(output="train.log", name=name, color=False)
                logger.info("hello from a bare log filename")
                for h in logger.handlers:
                    if hasattr(h, "flush"):
                        h.flush()
                self.assertTrue(os.path.isfile("train.log"))
                with open("train.log") as f:
                    self.assertIn("hello from a bare log filename", f.read())
            finally:
                os.chdir(cwd)
                logging.getLogger(name).handlers.clear()

    def test_setup_logger_nested_filename(self):
        """Parent directories of a nested log path are still created."""
        with tempfile.TemporaryDirectory(prefix="detectron2_tests") as tmp_dir:
            name = "detectron2_tests_logger_" + uuid.uuid4().hex
            path = os.path.join(tmp_dir, "logs", "train.log")
            try:
                logger = setup_logger(output=path, name=name, color=False)
                logger.info("hello from a nested log filename")
                for h in logger.handlers:
                    if hasattr(h, "flush"):
                        h.flush()
                self.assertTrue(os.path.isfile(path))
                with open(path) as f:
                    self.assertIn("hello from a nested log filename", f.read())
            finally:
                logging.getLogger(name).handlers.clear()
