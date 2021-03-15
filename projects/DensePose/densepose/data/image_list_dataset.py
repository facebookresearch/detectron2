# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
from typing import Callable, List, Optional
import torch
from torch.utils.data.dataset import Dataset

from detectron2.data.detection_utils import read_image

ImageTransform = Callable[[torch.Tensor], torch.Tensor]


class ImageListDataset(Dataset):
    """
    Dataset that provides images from a list.
    """

    _EMPTY_IMAGE = torch.empty((1, 1, 3))

    def __init__(self, image_list: List[str], transform: Optional[ImageTransform] = None):
        """
        Args:
            image_list (List[str]): list of paths to image files
        """
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Gets selected images from the list

        Args:
            idx (int): video index in the video list file
        Returns:
            image (torch.Tensor): tensor of size [H, W, 3]
        """
        fpath = self.image_list[idx]

        try:
            image = torch.from_numpy(np.ascontiguousarray(read_image(fpath, format="BGR")))
            if self.transform is not None:
                # Transforms are done on batches
                image = self.transform(image.unsqueeze(0))[0]  # pyre-ignore[29]
            return image
        except (OSError, RuntimeError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error opening image file container {fpath}: {e}")

        return self._EMPTY_IMAGE

    def __len__(self):
        return len(self.image_list)
