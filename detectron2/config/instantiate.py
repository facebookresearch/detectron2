# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses
import logging
from collections import abc
from typing import Any
from omegaconf import DictConfig

from detectron2.utils.registry import _convert_target_to_string, locate

__all__ = ["dump_dataclass", "instantiate"]


def dump_dataclass(obj: Any):
    """
    Dump a dataclass recursively into a dict that can be later instantiated.

    Args:
        obj: a dataclass object

    Returns:
        dict
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(
        obj, type
    ), "dump_dataclass() requires an instance of a dataclass."
    ret = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        ret[f.name] = v
    return ret


def instantiate(cfg):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    from omegaconf import ListConfig

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except AttributeError:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            return cls(**cfg)
        except TypeError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be execued,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Examples:
    ::
        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                "target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, **kwargs):
        kwargs["_target_"] = self._target
        return DictConfig(content=kwargs, flags={"allow_objects": True})
