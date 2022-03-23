# Copyright (c) Facebook, Inc. and its affiliates.

import torch


class ImageResizeTransform:
    """
    Transform that converts frames loaded from a dataset
    (RGB data in NHWC channel order, typically uint8) to a format ready to be
    consumed by DensePose training (BGR float32 data in NCHW channel order)
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (torch.Tensor): tensor of size [N, H, W, 3] that contains
                RGB data (typically in uint8)
        Returns:
            frames (torch.Tensor): tensor of size [N, 3, H1, W1] where
                H1 and W1 are chosen to respect the specified min and max sizes
                and preserve the original aspect ratio, the data channels
                follow BGR order and the data type is `torch.float32`
        """
        frames = frames[..., [2, 1, 0]]  # RGB -> BGR
        frames = frames.permute(0, 3, 1, 2).float()  # NHWC -> NCHW
        # resize with min size
        min_size = min(frames.shape[-2:])
        max_size = max(frames.shape[-2:])
        scale = min(self.min_size / min_size, self.max_size / max_size)
        frames = torch.nn.functional.interpolate(
            frames, scale_factor=scale, mode="bilinear", align_corners=False
        )
        return frames
