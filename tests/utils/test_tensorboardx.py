import os
import tempfile
import unittest

from detectron2.utils.events import TensorBoardXWriter


class TestTensorBoardXWriter(unittest.TestCase):
    def test_no_files_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = TensorBoardXWriter(tmp_dir)
            writer.close()

            self.assertFalse(os.listdir(tmp_dir))

    def test_single_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = TensorBoardXWriter(tmp_dir)
            writer._writer.add_scalar("testing", 1, 1)
            writer.close()

            self.assertTrue(os.listdir(tmp_dir))
