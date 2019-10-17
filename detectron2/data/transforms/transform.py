# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transform.py

import numpy as np
from fvcore.transforms.transform import HFlipTransform, NoOpTransform, Transform
from PIL import Image

__all__ = ["ExtentTransform", "ResizeTransform"]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
NoOpTransform.register_type("rotated_box", lambda t, x: x)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)
