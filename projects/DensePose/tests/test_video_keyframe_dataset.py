# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import os
import random
import tempfile
import unittest
import torch
import torchvision.io as io

from densepose.data.transform import ImageResizeTransform
from densepose.data.video import RandomKFramesSelector, VideoKeyframeDataset

try:
    import av
except ImportError:
    av = None


# copied from torchvision test/test_io.py
def _create_video_frames(num_frames, height, width):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())
    return torch.stack(data, 0)


# adapted from torchvision test/test_io.py
@contextlib.contextmanager
def temp_video(num_frames, height, width, fps, lossless=False, video_codec=None, options=None):
    if lossless:
        if video_codec is not None:
            raise ValueError("video_codec can't be specified together with lossless")
        if options is not None:
            raise ValueError("options can't be specified together with lossless")
        video_codec = "libx264rgb"
        options = {"crf": "0"}
    if video_codec is None:
        video_codec = "libx264"
    if options is None:
        options = {}
    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data
    os.unlink(f.name)


@unittest.skipIf(av is None, "PyAV unavailable")
class TestVideoKeyframeDataset(unittest.TestCase):
    def test_read_keyframes_all(self):
        with temp_video(60, 300, 300, 5, video_codec="mpeg4") as (fname, data):
            video_list = [fname]
            dataset = VideoKeyframeDataset(video_list)
            self.assertEqual(len(dataset), 1)
            data1 = dataset[0]
            self.assertEqual(data1.shape, torch.Size((5, 300, 300, 3)))
            self.assertEqual(data1.dtype, torch.uint8)
            return
        self.assertTrue(False)

    def test_read_keyframes_with_selector(self):
        with temp_video(60, 300, 300, 5, video_codec="mpeg4") as (fname, data):
            video_list = [fname]
            random.seed(0)
            frame_selector = RandomKFramesSelector(3)
            dataset = VideoKeyframeDataset(video_list, frame_selector)
            self.assertEqual(len(dataset), 1)
            data1 = dataset[0]
            self.assertEqual(data1.shape, torch.Size((3, 300, 300, 3)))
            self.assertEqual(data1.dtype, torch.uint8)
            return
        self.assertTrue(False)

    def test_read_keyframes_with_selector_with_transform(self):
        with temp_video(60, 300, 300, 5, video_codec="mpeg4") as (fname, data):
            video_list = [fname]
            random.seed(0)
            frame_selector = RandomKFramesSelector(1)
            transform = ImageResizeTransform()
            dataset = VideoKeyframeDataset(video_list, frame_selector, transform)
            data1 = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(data1.shape, torch.Size((1, 3, 800, 800)))
            self.assertEqual(data1.dtype, torch.float32)
            return
        self.assertTrue(False)
