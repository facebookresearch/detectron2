# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
from typing import IO, Callable, List, Optional
import torch
from fvcore.common.file_io import PathManager
from torch.utils.data.dataset import Dataset

import av

from ..utils import maybe_prepend_base_path
from .frame_selector import FrameSelector, FrameTsList

FrameList = List[av.frame.Frame]
FrameTransform = Callable[[torch.Tensor], torch.Tensor]


def _list_keyframes(
    video_fpath: str, video_file: IO[bytes], video_stream_idx: int = 0
) -> FrameTsList:
    """
    Traverses all keyframes of a video file. Returns a list of keyframe
    timestamps. Timestamps are counts in timebase units.

    Args:
       video_fpath (str): Video file path
       video_file (IO[bytes]): Video file input stream
       video_stream_idx (int): Video stream index (default: 0)
    Returns:
       List[int]: list of keyframe timestaps (timestamp is a count in timebase
           units)
    """
    container = av.open(video_file)
    s = container.streams[video_stream_idx]
    keyframes = []
    pts = -1
    while True:
        try:
            container.seek(pts + 1, backward=False, any_frame=False, stream=s)
        except av.AVError:
            break
        packet = next(container.demux(video=video_stream_idx))
        if packet.pts is not None and packet.pts <= pts:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Video file {video_fpath}, stream {video_stream_idx}: "
                f"bad seek for packet {pts} (got packet {packet.pts}), "
                f"returning empty keyframes."
            )
            return []
        pts = packet.pts
        if pts is None:
            return keyframes
        if packet.is_keyframe:
            keyframes.append(pts)
    return keyframes


def _read_keyframes(video_file: IO[bytes], keyframes: FrameTsList) -> FrameList:
    """
    Reads keyframe data from a video file.

    Args:
        video_file (IO[bytes]): Opened file in binary mode
        keyframes (List[int]): List of keyframe timestamps (as counts in
            timebase units to be used in container seek operations)
    """
    container = av.open(video_file)
    frames = []
    for pts in keyframes:
        container.seek(pts, any_frame=False, stream=container.streams[0])
        frame = next(container.decode(video=0))
        frames.append(frame)
    container.close()
    return frames


def video_list_from_file(video_list_fpath: str, base_path: Optional[str] = None):
    """
    Create a list of paths to video files from a text file.

    Args:
        video_list_fpath (str): path to a plain text file with the list of videos
        base_path (str): base path for entries from the video list (default: None)
    """
    video_list = []
    with PathManager.open(video_list_fpath, "r") as io:
        for line in io:
            video_list.append(maybe_prepend_base_path(line, base_path))
    return video_list


class VideoKeyframeDataset(Dataset):
    """
    Dataset that provides keyframes for a set of videos.
    """

    _EMPTY_FRAMES = torch.empty((0, 3, 1, 1))

    def __init__(
        self,
        video_list: List[str],
        frame_selector: Optional[FrameSelector] = None,
        transform: Optional[FrameTransform] = None,
    ):
        """
        Dataset constructor

        Args:
            video_list (List[str]): list of paths to video files
            frame_selector (Callable: KeyFrameList -> KeyFrameList):
                selects keyframes to process, keyframes are given by
                packet timestamps in timebase counts. If None, all keyframes
                are selected (default: None)
            transform (Callable: torch.Tensor -> torch.Tensor):
                transforms a batch of RGB images (tensors of size [B, H, W, 3]),
                returns a tensor of the same size. If None, no transform is
                applied (default: None)

        """
        self.video_list = video_list
        self.frame_selector = frame_selector
        self.transform = transform

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Gets selected keyframes from a given video

        Args:
            idx (int): video index in the video list file
        Returns:
            frames (torch.Tensor): tensor of size [N, H, W, 3] or of size
                defined by the transform that contains keyframes data
        """
        fpath = self.video_list[idx]
        with PathManager.open(fpath, "rb") as hFile:
            keyframes = _list_keyframes(fpath, hFile)
        if not keyframes:
            return self.EMPTY_FRAMES
        if self.frame_selector is not None:
            keyframes = self.frame_selector(keyframes)
        with PathManager.open(fpath, "rb") as hFile:
            frames = _read_keyframes(hFile, keyframes)
        if not frames:
            return self.EMPTY_FRAMES
        frames = np.stack([frame.to_rgb().to_ndarray() for frame in frames])
        frames = torch.as_tensor(frames)
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    def __len__(self):
        return len(self.video_list)
