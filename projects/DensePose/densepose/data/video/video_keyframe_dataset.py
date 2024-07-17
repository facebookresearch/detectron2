# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

import csv
import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import av
import torch
from torch.utils.data.dataset import Dataset

from detectron2.utils.file_io import PathManager

from ..utils import maybe_prepend_base_path
from .frame_selector import FrameSelector, FrameTsList

FrameList = List[av.frame.Frame]  # pyre-ignore[16]
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
            # pyre-fixme[16]: Module `av` has no attribute `open`.
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
) -> FrameList:  # pyre-ignore[11]
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
            # pyre-fixme[16]: Module `av` has no attribute `open`.
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
            video_list.append(maybe_prepend_base_path(base_path, str(line.strip())))
    return video_list


def read_keyframe_helper_data(fpath: str):
    """
    Read keyframe data from a file in CSV format: the header should contain
    "video_id" and "keyframes" fields. Value specifications are:
      video_id: int
      keyframes: list(int)
    Example of contents:
      video_id,keyframes
      2,"[1,11,21,31,41,51,61,71,81]"

    Args:
        fpath (str): File containing keyframe data

    Return:
        video_id_to_keyframes (dict: int -> list(int)): for a given video ID it
          contains a list of keyframes for that video
    """
    video_id_to_keyframes = {}
    try:
        with PathManager.open(fpath, "r") as io:
            csv_reader = csv.reader(io)
            header = next(csv_reader)
            video_id_idx = header.index("video_id")
            keyframes_idx = header.index("keyframes")
            for row in csv_reader:
                video_id = int(row[video_id_idx])
                assert (
                    video_id not in video_id_to_keyframes
                ), f"Duplicate keyframes entry for video {fpath}"
                video_id_to_keyframes[video_id] = (
                    [int(v) for v in row[keyframes_idx][1:-1].split(",")]
                    if len(row[keyframes_idx]) > 2
                    else []
                )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error reading keyframe helper data from {fpath}: {e}")
    return video_id_to_keyframes


class VideoKeyframeDataset(Dataset):
    """
    Dataset that provides keyframes for a set of videos.
    """

    _EMPTY_FRAMES = torch.empty((0, 3, 1, 1))

    def __init__(
        self,
        video_list: List[str],
        category_list: Union[str, List[str], None] = None,
        frame_selector: Optional[FrameSelector] = None,
        transform: Optional[FrameTransform] = None,
        keyframe_helper_fpath: Optional[str] = None,
    ):
        """
        Dataset constructor

        Args:
            video_list (List[str]): list of paths to video files
            category_list (Union[str, List[str], None]): list of animal categories for each
                video file. If it is a string, or None, this applies to all videos
            frame_selector (Callable: KeyFrameList -> KeyFrameList):
                selects keyframes to process, keyframes are given by
                packet timestamps in timebase counts. If None, all keyframes
                are selected (default: None)
            transform (Callable: torch.Tensor -> torch.Tensor):
                transforms a batch of RGB images (tensors of size [B, 3, H, W]),
                returns a tensor of the same size. If None, no transform is
                applied (default: None)

        """
        if type(category_list) is list:
            self.category_list = category_list
        else:
            self.category_list = [category_list] * len(video_list)
        assert len(video_list) == len(
            self.category_list
        ), "length of video and category lists must be equal"
        self.video_list = video_list
        self.frame_selector = frame_selector
        self.transform = transform
        self.keyframe_helper_data = (
            read_keyframe_helper_data(keyframe_helper_fpath)
            if keyframe_helper_fpath is not None
            else None
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets selected keyframes from a given video

        Args:
            idx (int): video index in the video list file
        Returns:
            A dictionary containing two keys:
                images (torch.Tensor): tensor of size [N, H, W, 3] or of size
                    defined by the transform that contains keyframes data
                categories (List[str]): categories of the frames
        """
        categories = [self.category_list[idx]]
        fpath = self.video_list[idx]
        keyframes = (
            list_keyframes(fpath)
            if self.keyframe_helper_data is None or idx not in self.keyframe_helper_data
            else self.keyframe_helper_data[idx]
        )
        transform = self.transform
        frame_selector = self.frame_selector
        if not keyframes:
            return {"images": self._EMPTY_FRAMES, "categories": []}
        if frame_selector is not None:
            keyframes = frame_selector(keyframes)
        frames = read_keyframes(fpath, keyframes)
        if not frames:
            return {"images": self._EMPTY_FRAMES, "categories": []}
        frames = np.stack([frame.to_rgb().to_ndarray() for frame in frames])
        frames = torch.as_tensor(frames, device=torch.device("cpu"))
        frames = frames[..., [2, 1, 0]]  # RGB -> BGR
        frames = frames.permute(0, 3, 1, 2).float()  # NHWC -> NCHW
        if transform is not None:
            frames = transform(frames)
        return {"images": frames, "categories": categories}

    def __len__(self):
        return len(self.video_list)
