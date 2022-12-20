# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
from detectron2.data.transforms import Augmentation, Transform, ResizeTransform
from PIL import Image
from fvcore.transforms.transform import NoOpTransform
import cv2


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


class EraseTransform(Transform):
    def __init__(self, patches: list, fill_val: int):
        super().__init__()
        self.patches = patches
        self.fill_val = fill_val

    def apply_image(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy())
        for pat in self.patches:
            x1, y1, x2, y2 = pat
            img[y1:y2, x1:x2, :] = self.fill_val
        return img.numpy()

    def apply_coords(self, coords: np.ndarray):
        return coords

    def apply_segmentation(self, segmentation: np.ndarray):
        return segmentation


class RandErase(Augmentation):
    def __init__(
            self,
            n_iterations=(1, 5),
            size=None,
            squared: bool = True,
            patches=None,
            img_fill_val=125,
            random_magnitude=True,
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches
        self.img_fill_val = img_fill_val
        self.random_magnitude = random_magnitude

    def get_transform(self, image, instances):
        h, w = image.shape[:2]
        # get magnitude
        patches = []
        if self.random_magnitude:
            for i, bbox in enumerate(instances.gt_boxes):
                x1, y1, x2, y2 = bbox.int()
                if x1 != x2 and y1 != y2:
                    if instances.gt_densepose[i] is not None:
                        n_iterations = self._get_erase_cycle()
                    else:
                        n_iterations = torch.randint(1, 2, (1,))
                    for _ in range(n_iterations):
                        ph, pw = self._get_patch_size(y2 - y1, x2 - x1)
                        px, py = torch.randint(x1, x2, (1,)).clamp(0, w - pw), torch.randint(y1, y2, (1,)).clamp(0, h - ph)
                        patches.append([px, py, px + pw, py + ph])
        else:
            assert self.patches is not None
            patches = self.patches

        return EraseTransform(patches, self.img_fill_val)

    # def __call__(self, inputs):
    #     if torch.rand((1, )) < self.prob:
    #         for ipt in inputs:
    #             magnitude: dict = self.get_magnitude(ipt)
    #             ipt['image'] = self.apply(ipt, **magnitude)
    #
    #     return inputs

    # def get_magnitude(self, h, w):
    #     magnitude = {}
    #     if self.random_magnitude:
    #         n_iterations = self._get_erase_cycle()
    #         patches = []
    #         for _ in range(n_iterations):
    #             # random sample patch size in the image
    #             ph, pw = self._get_patch_size(h, w)
    #             # random sample patch left top in the image
    #             # px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
    #             px, py = torch.randint(0, w - pw, (1,)), torch.randint(0, h - ph, (1,))
    #             patches.append([px, py, px + pw, py + ph])
    #         magnitude["patches"] = patches
    #     else:
    #         assert self.patches is not None
    #         magnitude["patches"] = self.patches
    #
    #     return magnitude

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
                        torch.rand((1,)) * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    # def apply(self, ipt, patches: list):
    #     image = ipt['image']
    #     for pat in patches:
    #         image = erase_image(image, pat, fill_val=self.img_fill_val)
    #         # self._erase_mask(inputs, patch)
    #         # self._erase_seg(inputs, patch, fill_val=self.seg_ignore_label)
    #     return image

class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        w, h = box[0, 2] - box[0, 0], box[0, 3] - box[0, 1]
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        # minxy = coords.min(axis=1)
        # maxxy = coords.max(axis=1)
        wh = np.array([[w, h]]) / 2
        lxy = coords.mean(axis=1) - wh
        rxy = coords.mean(axis=1) + wh
        trans_boxes = np.concatenate((lxy, rxy), axis=1)
        return trans_boxes

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])
