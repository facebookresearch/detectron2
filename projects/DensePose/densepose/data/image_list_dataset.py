# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data.dataset import Dataset

from detectron2.data.detection_utils import read_image

ImageTransform = Callable[[torch.Tensor], torch.Tensor]


class ImageListDataset(Dataset):
    """
    Dataset that provides images from a list.
    """

    _EMPTY_IMAGE = torch.empty((0, 3, 1, 1))

    def __init__(
        self,
        image_list: List[str],
        category_list: Union[str, List[str], None] = None,
        transform: Optional[ImageTransform] = None,
    ):
        """
        Args:
            image_list (List[str]): list of paths to image files
            category_list (Union[str, List[str], None]): list of animal categories for
                each image. If it is a string, or None, this applies to all images
        """
        if type(category_list) is list:
            self.category_list = category_list
        else:
            self.category_list = [category_list] * len(image_list)
        assert len(image_list) == len(
            self.category_list
        ), "length of image and category lists must be equal"
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets selected images from the list

        Args:
            idx (int): video index in the video list file
        Returns:
            A dictionary containing two keys:
                images (torch.Tensor): tensor of size [N, 3, H, W] (N = 1, or 0 for _EMPTY_IMAGE)
                categories (List[str]): categories of the frames
        """
        categories = [self.category_list[idx]]
        fpath = self.image_list[idx]
        transform = self.transform

        try:
            image = torch.from_numpy(np.ascontiguousarray(read_image(fpath, format="BGR")))
            image = image.permute(2, 0, 1).unsqueeze(0).float()  # HWC -> NCHW
            if transform is not None:
                image = transform(image)
            return {"images": image, "categories": categories}
        except (OSError, RuntimeError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error opening image file container {fpath}: {e}")

        return {"images": self._EMPTY_IMAGE, "categories": []}

    def __len__(self):
        return len(self.image_list)
