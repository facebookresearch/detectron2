# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import division
from typing import Any, List, Tuple
import torch
from torch import device
from torch.nn import functional as F

from detectron2.utils.env import TORCH_VERSION


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape

        # Magic code below that handles dynamic shapes for both scripting and tracing ...

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]

        if torch.jit.is_scripting():
            max_size = torch.stack([torch.as_tensor(x) for x in image_sizes]).max(0).values
            if size_divisibility > 1:
                stride = size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size = (max_size + (stride - 1)) // stride * stride

            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            # https://github.com/pytorch/pytorch/issues/42448
            if TORCH_VERSION >= (1, 7) and torch.jit.is_tracing():
                # In tracing mode, x.shape[i] is a scalar Tensor, and should not be converted
                # to int: this will cause the traced graph to have hard-coded shapes.
                # Instead we convert each shape to a vector with a stack()
                image_sizes = [torch.stack(x) for x in image_sizes]

                # maximum (H, W) for the last two dims
                # find the maximum in a tracable way
                max_size = torch.stack(image_sizes).max(0).values
            else:
                # Original eager logic here -- not scripting, not tracing:
                # (can be unified with scripting after
                # https://github.com/pytorch/pytorch/issues/47379)
                max_size = torch.as_tensor(
                    [max(s) for s in zip(*[img.shape[-2:] for img in tensors])]
                )

            if size_divisibility > 1:
                stride = size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size = (max_size + (stride - 1)) // stride * stride

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)
