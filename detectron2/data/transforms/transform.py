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
    Apply the horizontal flip transform on an rotated boxes.

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
    # Note: when scale_factor_x != scale_factor_y,
    # the rotated box does not preserve the rectangular shape when the angle
    # is not a multiple of 90 degrees under resize transformation.
    # Instead, the shape is a parallelogram (that has skew)
    # Here we make an approximation by fitting a rotated rectangle to the
    # parallelogram that shares the same midpoints on the left and right edge
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)

    # In image space, y is top->down and x is left->right
    # Consider the local coordinate system for the rotated box,
    # where the box center is located at (0, 0), and the four vertices ABCD are
    # A(-w / 2, -h / 2), B(w / 2, -h / 2), C(w / 2, h / 2), D(-w / 2, h / 2)
    # the midpoint of the left edge AD of the rotated box E is:
    # E = (A+D)/2 = (-w / 2, 0)
    # the midpoint of the top edge AB of the rotated box F is:
    # F(0, -h / 2)
    # To get the old coordinates in the global system, apply the rotation transformation
    # (Note: the right-handed coordinate system for image space is yOx):
    # (old_x, old_y) = (s * y + c * x, c * y - s * x)
    # E(old) = (s * 0 + c * (-w/2), c * 0 - s * (-w/2)) = (-c * w / 2, s * w / 2)
    # F(old) = (s * (-h / 2) + c * 0, c * (-h / 2) - s * 0) = (-s * h / 2, -c * h / 2)
    # After applying the scaling factor (sfx, sfy):
    # E(new) = (-sfx * c * w / 2, sfy * s * w / 2)
    # F(new) = (-sfx * s * h / 2, -sfy * c * h / 2)
    # The new width after scaling transformation becomes:

    # w(new) = |E(new) - O| * 2
    #        = sqrt[(sfx * c * w / 2)^2 + (sfy * s * w / 2)^2] * 2
    #        = sqrt[(sfx * c)^2 + (sfy * s)^2] * w
    # i.e., scale_factor_w = sqrt[(sfx * c)^2 + (sfy * s)^2]
    #
    # For example,
    # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_w == scale_factor_x;
    # when |angle| = 90, c = 0, |s| = 1, scale_factor_w == scale_factor_y
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))

    # h(new) = |F(new) - O| * 2
    #        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
    #        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
    # i.e., scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
    #
    # For example,
    # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_h == scale_factor_y;
    # when |angle| = 90, c = 0, |s| = 1, scale_factor_h == scale_factor_x
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))

    # The angle is the rotation angle from y-axis in image space to the height
    # vector (top->down in the box's local coordinate system) of the box in CCW.
    #
    # angle(new) = angle_yOx(O - F(new))
    #            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
    #            = atan2(sfx * s * h / 2, sfy * c * h / 2)
    #            = atan2(sfx * s, sfy * c)
    #
    # For example,
    # when sfx == sfy, angle(new) == atan2(s, c) == angle(old)
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
NoOpTransform.register_type("rotated_box", lambda t, x: x)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)
