# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any
import pydoc
from fvcore.common.registry import Registry  # for backward compatibility.

"""
``Registry`` and `locate` provide ways to map a string (typically found
in config files) to callable objects.
"""

__all__ = ["Registry", "locate"]


def _convert_target_to_string(t: Any) -> Any:
    """
    Inverse of ``locate()``.
    """
    return f"{t.__module__}.{t.__qualname__}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    # Should use _locate directly if it's public.
    if obj is None:
        try:
            from hydra.utils import get_method
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = get_method(name)  # it raises if fails

    return obj
