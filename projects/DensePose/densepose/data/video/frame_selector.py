# Copyright (c) Facebook, Inc. and its affiliates.

import random
from collections.abc import Callable
from enum import Enum
from typing import Callable as TCallable
from typing import List

FrameTsList = List[int]
FrameSelector = TCallable[[FrameTsList], FrameTsList]


class FrameSelectionStrategy(Enum):
    """
    Frame selection strategy used with videos:
     - "random_k": select k random frames
     - "first_k": select k first frames
     - "last_k": select k last frames
     - "all": select all frames
    """

    # fmt: off
    RANDOM_K = "random_k"
    FIRST_K  = "first_k"
    LAST_K   = "last_k"
    ALL      = "all"
    # fmt: on


class RandomKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` random frames
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` random frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return random.sample(frame_tss, min(self.k, len(frame_tss)))


class FirstKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` first frames
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` first frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return frame_tss[: self.k]


class LastKFramesSelector(Callable):  # pyre-ignore[39]
    """
    Selector that retains at most `k` last frames from video data
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, frame_tss: FrameTsList) -> FrameTsList:
        """
        Select `k` last frames

        Args:
          frames_tss (List[int]): timestamps of input frames
        Returns:
          List[int]: timestamps of selected frames
        """
        return frame_tss[-self.k :]
