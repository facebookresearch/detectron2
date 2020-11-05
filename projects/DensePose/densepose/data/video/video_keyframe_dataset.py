# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
from typing import Callable, List, Optional
import torch
from torch.utils.data.dataset import Dataset

from detectron2.utils.file_io import PathManager

import av

from ..utils import maybe_prepend_base_path
from .frame_selector import FrameSelector, FrameTsList

FrameList = List[av.frame.Frame]
FrameTransform = Callable[[torch.Tensor], torch.Tensor]


def list_keyframes(video_fpath: str, video_stream_idx: int = 0) -> FrameTsList:
    """
    Traverses all keyframes of a video file. Returns a list of keyframe
    timestamps. Timestamps are counts in timebase units.

    Args:
       video_fpath (str): Video file path
       video_stream_idx (int): Video stream index (default: 0)
    Returns:
       List[int]: list of keyframe timestaps (timestamp is a count in timebase
           units)
    """
    try:
        with PathManager.open(video_fpath, "rb") as io:
            container = av.open(io, mode="r")
            stream = container.streams.video[video_stream_idx]
            keyframes = []
            pts = -1
            # Note: even though we request forward seeks for keyframes, sometimes
            # a keyframe in backwards direction is returned. We introduce tolerance
            # as a max count of ignored backward seeks
            tolerance_backward_seeks = 2
            while True:
                try:
                    container.seek(pts + 1, backward=False, any_frame=False, stream=stream)
                except av.AVError as e:
                    # the exception occurs when the video length is exceeded,
                    # we then return whatever data we've already collected
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"List keyframes: Error seeking video file {video_fpath}, "
                        f"video stream {video_stream_idx}, pts {pts + 1}, AV error: {e}"
                    )
                    return keyframes
                except OSError as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"List keyframes: Error seeking video file {video_fpath}, "
                        f"video stream {video_stream_idx}, pts {pts + 1}, OS error: {e}"
                    )
                    return []
                packet = next(container.demux(video=video_stream_idx))
                if packet.pts is not None and packet.pts <= pts:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Video file {video_fpath}, stream {video_stream_idx}: "
                        f"bad seek for packet {pts + 1} (got packet {packet.pts}), "
                        f"tolerance {tolerance_backward_seeks}."
                    )
                    tolerance_backward_seeks -= 1
                    if tolerance_backward_seeks == 0:
                        return []
                    pts += 1
                    continue
                tolerance_backward_seeks = 2
                pts = packet.pts
                if pts is None:
                    return keyframes
                if packet.is_keyframe:
                    keyframes.append(pts)
            return keyframes
    except OSError as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"List keyframes: Error opening video file container {video_fpath}, " f"OS error: {e}"
        )
    except RuntimeError as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"List keyframes: Error opening video file container {video_fpath}, "
            f"Runtime error: {e}"
        )
    return []


def read_keyframes(
    video_fpath: str, keyframes: FrameTsList, video_stream_idx: int = 0
) -> FrameList:
    """
    Reads keyframe data from a video file.

    Args:
        video_fpath (str): Video file path
        keyframes (List[int]): List of keyframe timestamps (as counts in
            timebase units to be used in container seek operations)
        video_stream_idx (int): Video stream index (default: 0)
    Returns:
        List[Frame]: list of frames that correspond to the specified timestamps
    """
    try:
        with PathManager.open(video_fpath, "rb") as io:
            container = av.open(io)
            stream = container.streams.video[video_stream_idx]
            frames = []
            for pts in keyframes:
                try:
                    container.seek(pts, any_frame=False, stream=stream)
                    frame = next(container.decode(video=0))
                    frames.append(frame)
                except av.AVError as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Read keyframes: Error seeking video file {video_fpath}, "
                        f"video stream {video_stream_idx}, pts {pts}, AV error: {e}"
                    )
                    container.close()
                    return frames
                except OSError as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Read keyframes: Error seeking video file {video_fpath}, "
                        f"video stream {video_stream_idx}, pts {pts}, OS error: {e}"
                    )
                    container.close()
                    return frames
                except StopIteration:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Read keyframes: Error decoding frame from {video_fpath}, "
                        f"video stream {video_stream_idx}, pts {pts}"
                    )
                    container.close()
                    return frames

            container.close()
            return frames
    except OSError as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Read keyframes: Error opening video file container {video_fpath}, OS error: {e}"
        )
    except RuntimeError as e:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Read keyframes: Error opening video file container {video_fpath}, Runtime error: {e}"
        )
    return []


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
            video_list.append(maybe_prepend_base_path(base_path, line.strip()))
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
        keyframes = list_keyframes(fpath)
        if not keyframes:
            return self._EMPTY_FRAMES
        if self.frame_selector is not None:
            keyframes = self.frame_selector(keyframes)
        frames = read_keyframes(fpath, keyframes)
        if not frames:
            return self._EMPTY_FRAMES
        frames = np.stack([frame.to_rgb().to_ndarray() for frame in frames])
        frames = torch.as_tensor(frames, device=torch.device("cpu"))
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    def __len__(self):
        return len(self.video_list)
