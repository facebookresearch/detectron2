# Copyright (c) Facebook, Inc. and its affiliates.
from requests import patch
import torch
from torch.nn import functional as F
import numpy as np


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
            # pyre-fixme[6]: Expected `Optional[typing.List[float]]` for 2nd param
            #  but got `float`.
            images, scale_factor=scale, mode="bilinear", align_corners=False
        )
        return images


def erase_image(image, pat, fill_val=128):
    x1, y1, x2, y2 = pat
    image[:, y1:y2, x1:x2] = fill_val
    return image


class RandErase:
    def __init__(
        self,
        n_iterations=(1, 5),
        size=None,
        squared: bool = True,
        patches=None,
        img_fill_val=125,
        random_magnitude=True,
        prob=1.0
    ) -> None:
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches
        self.img_fill_val = img_fill_val
        self.random_magnitude = random_magnitude
        self.prob = prob

    def __call__(self, inputs):
        if torch.rand((1, )) < self.prob:
            for ipt in inputs:
                magnitude: dict = self.get_magnitude(ipt)
                ipt['image'] = self.apply(ipt, **magnitude)

        return inputs

    def get_magnitude(self, ipt):
        magnitude = {}
        if self.random_magnitude:
            n_iterations = self._get_erase_cycle()
            patches = []
            _, h, w = ipt["image"].shape
            for _ in range(n_iterations):
                # random sample patch size in the image
                ph, pw = self._get_patch_size(h, w)
                # random sample patch left top in the image
                # px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
                px, py = torch.randint(0, w - pw, (1,)), torch.randint(0, h - ph, (1,))
                patches.append([px, py, px + pw, py + ph])
            magnitude["patches"] = patches
        else:
            assert self.patches is not None
            magnitude["patches"] = self.patches

        return magnitude

    def _get_erase_cycle(self):
        if isinstance(self.n_iterations, int):
            n_iterations = self.n_iterations
        else:
            assert (
                isinstance(self.n_iterations, (tuple, list))
                and len(self.n_iterations) == 2
            )
            # n_iterations = np.random.randint(*self.n_iterations)
            n_iterations = torch.randint(self.n_iterations[0], self.n_iterations[1], (1,))

        return n_iterations

    def _get_patch_size(self, h, w):
        if isinstance(self.size, float):
            assert 0 < self.size < 1
            return int(self.size * h), int(self.size * w)
        else:
            assert isinstance(self.size, (tuple, list))
            assert len(self.size) == 2
            assert 0 <= self.size[0] < 1 and 0 <= self.size[1] < 1
            # w_ratio = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
            w_ratio = torch.rand((1,)) * (self.size[1] - self.size[0]) + self.size[0]
            h_ratio = w_ratio

            if not self.squared:
                h_ratio = (
                    # np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
                    torch.rand((1, )) * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    def apply(self, ipt, patches: list):
        image = ipt['image']
        for pat in patches:
            image = erase_image(image, pat, fill_val=self.img_fill_val)
            # self._erase_mask(inputs, patch)
            # self._erase_seg(inputs, patch, fill_val=self.seg_ignore_label)
        return image


class RandomResize:
    def __init__(
        self,
        short_edge_length,
        max_size,
        sample_styple='choice',
        interp='bilinear'
    ):
        self.is_range = sample_styple == 'range'
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        
        self.short_edge_length = short_edge_length
        self.max_size = max_size,
        self.sample_stype = sample_styple
        self.interp = interp

    def __call__(self, inputs):
        if torch.rand((1, )) < self.prob:
            for ipt in inputs:
                # get new h and w
                hw = ipt['image'].shape[:-2]
                newhw = self.get_output_shape(hw)
                # resize the image
                ipt['image'] = self.apply_image(ipt['image'], newhw)
                # transform the boxes
                ipt['gt_box'] = self.apply_box(ipt['gt_box'], newhw, hw)
                # Resize dont neet to transform densepose annotataions
    
    def apply_image(self, image, newhw):
        newh, neww = newhw
        align_corners = None if self.interp == 'nearest' else False
        return F.interpolate(image, (newh, neww), mode=self.interp, align_corners=align_corners)

    def apply_box(self, boxes, newhw, hw):
        newh, neww = newhw
        h, w = hw
        boxes = boxes[:, [0,2,1,3]]
        boxes[:, :2] = boxes[:, :2] * (neww * 1.0 / w)
        boxes[:, -2:] = boxes[:, -2:] * (newh * 1.0 / h)
        return boxes[:, [0,2,1,3]]

    def get_output_shape(self, hw):
        h, w = hw
        if self.is_range:
            size = torch.randint(self.short_edge_length[0], self.short_edge_length[1] + 1, (1,))
        else:
            size = torch.tensor(np.random.choice(self.short_edge_length))
        if size == 0:
            return None
        
        scale = size / self.max_size
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size

        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class RandomRotation:
    def __init__(
        self
    ) -> None:
        pass
