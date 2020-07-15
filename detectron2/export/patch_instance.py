# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib.util
import os
import sys
import tempfile
import torch

from detectron2.structures import Boxes, Instances  # noqa F401

_counter = 0  # TODO replace by a hash?


def export_torchscript_with_patch_instance(model, fields):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as f:
        cls_name, s = gen_class(fields)
        f.write(s)
        f.flush()
        f.close()

        module = _import(f.name)
        new_instance = getattr(module, cls_name)
        _ = torch.jit.script(new_instance)

        Instances.__torch_script_class__ = True
        Instances._jit_override_qualname = torch._jit_internal._qualified_name(new_instance)
        return torch.jit.script(model), new_instance


def _typing_imports_str():

    import_str = """
import torch
from torch import Tensor
import typing
from typing import *

from detectron2.structures import Boxes

"""
    return import_str


def _gen_class(fields):
    def indent(level, s):
        return " " * 4 * level + s

    lines = []

    global _counter
    _counter += 1

    cls_name = "Instance{}".format(_counter)

    lines.append(
        f"""

class {cls_name}:
    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size

"""
    )

    for name, type_ in fields.items():
        lines.append(
            indent(2, "self.{} = torch.jit.annotate(Optional[{}], None)".format(name, type_))
        )

    return cls_name, os.linesep.join(lines)


def gen_class(fields):
    s = ""
    s += _typing_imports_str()
    cls_name, cls_def = _gen_class(fields)
    s += cls_def
    return cls_name, s


def _import(path):
    spec = importlib.util.spec_from_file_location(
        "{}{}".format(sys.modules[__name__].__name__, _counter), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module
