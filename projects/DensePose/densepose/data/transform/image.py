# Copyright (c) Facebook, Inc. and its affiliates.

import torch


class ImageResizeTransform:
    """
    Transform that resizes images loaded from a dataset
    (BGR data in NCHW channel order, typically uint8) to a format ready to be
    consumed by DensePose training (BGR float32 data in NCHW channel order)
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): tensor of size [N, 3, H, W] that contains
                BGR data (typically in uint8)
        Returns:
            images (torch.Tensor): tensor of size [N, 3, H1, W1] where
                H1 and W1 are chosen to respect the specified min and max sizes
                and preserve the original aspect ratio, the data channels
                follow BGR order and the data type is `torch.float32`
        """
        # resize with min size
        images = images.float()
        min_size = min(images.shape[-2:])
        max_size = max(images.shape[-2:])
        scale = min(self.min_size / min_size, self.max_size / max_size)
        images = torch.nn.functional.interpolate(
            images, scale_factor=scale, mode="bilinear", align_corners=False
        )
        return images
