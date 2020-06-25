# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import inspect
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod
from fvcore.transforms.transform import Transform, TransformList

__all__ = ["Augmentation", "TransformGen", "apply_transform_gens", "apply_augmentations"]


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class Augmentation(metaclass=ABCMeta):
    """
    Augmentation takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `Augmentation` can be applied with :func:`apply_augmentations`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


TransformGen = Augmentation
"""
Alias for Augmentation, since it is something that generates :class:`Transform`s
"""


def apply_augmentations(augmentations, img):
    """
    Apply a list of :class:`Augmentation` or :class:`Transform` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        augmentations (list): list of :class:`Augmentation` or :class:`Transform` instance to
            be applied.
        img (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for aug in augmentations:
        assert isinstance(aug, (Transform, Augmentation)), aug

    check_dtype(img)

    tfms = []
    for aug in augmentations:
        tfm = aug.get_transform(img) if isinstance(aug, Augmentation) else aug
        assert isinstance(
            tfm, Transform
        ), f"Augmentation {aug} must return an instance of Transform! Got {tfm} instead."
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)


apply_transform_gens = apply_augmentations
"""
Alias for backward-compatibility.
"""
