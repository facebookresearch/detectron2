# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib.util
import os
import sys
import tempfile
import torch

from detectron2.structures import Boxes, Instances  # noqa F401


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
    lines = []
    lines.append("import torch")
    lines.append("from torch import Tensor")
    lines.append("import typing")
    lines.append("from typing import *")
    lines.append("")
    lines.append("")
    lines.append("from detectron2.structures import Boxes")
    lines.append("")
    lines.append("")
    return os.linesep.join(lines)


_counter = 0  # TODO replace by a hash?


def _gen_class(fields):
    def indent(level, s):
        return " " * 4 * level + s

    lines = []

    cls_name = "Instance{}".format(_counter)
    lines.append("class {}:".format(cls_name))
    lines.append("")

    lines.append(indent(1, "def __init__(self, image_size: Tuple[int, int]):"))
    lines.append(indent(2, "self.image_size = image_size"))
    for name, type_ in fields.items():
        lines.append(
            indent(2, "self.{} = torch.jit.annotate(Optional[{}], None)".format(name, type_))
        )
    lines.append("")

    for name, type_ in fields.items():
        # getter
        lines.append(indent(1, "def get_{}(self) -> {}:".format(name, type_)))
        lines.append(indent(2, "val = self.{}".format(name)))
        lines.append(indent(2, "assert val is not None"))
        lines.append(indent(2, "return val"))
        lines.append("")
        # setter
        lines.append(indent(1, "def set_{}(self, val: {}):".format(name, type_)))
        lines.append(indent(2, "self.{} = val".format(name)))
        lines.append("")

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
