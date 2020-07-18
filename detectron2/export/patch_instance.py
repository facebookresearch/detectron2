# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import importlib.util
import os
import sys
import tempfile
from contextlib import contextmanager
import torch

# https://github.com/pytorch/pytorch/issues/38964
from detectron2.structures import Boxes, Instances  # noqa F401

_counter = 0


def export_torchscript_with_patch_instance(model, fields):
    """
    Export the input model to torchscript with the data structure `Instances` been patched.
    Since Attributes of `Instances` is "dynamically" registered in the runtime，it may be impossible
    for torchscript to support it. To solve this problem, we do the following things:
        1) Use a series of strings to describe a torchscript-capable `new_Instances` with all
           attributes been "static". `new_Instances` have the same kinds of attributes with the
           corresponding old `Instances`
        2) Import the `new_Instances` and register it to torchscript.
        3) Patch the old `Instances` with `new_Instances`. So when torchscript is going to compile
           old `Instances` in the input model, it will use the `new_Instances` registered in the
           step 2.
    After the export process, the patch procedure will be reverted and the imported `new_Instances`
    class will be deleted, which make the above things have no side effects on other codes.

    Example:
        Assume that `Instances` in your model will consist of two attributes named `proposal_boxes`
        and `objectness_logits` with type of `Boxes` and `Tensor` respectively during the inference
        time. You can call this function like:
        ```python
            fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}
            torchscipt_model =  export_torchscript_with_patch_instance(model, fields)
        ```

    Args:
        model (torch.nn.Module): The input model to be exported to torchscript.
        fields (Dict[str, str]): Attribute names and corresponding annotations that `Instances'
            will contains during the inference time. Note that all attributes present in
            `Instances` need to be added, whether or not they are retained eventually.

    returns:
        scripted_model (torch.jit.ScriptModule): Torchscript of the input model.

    """
    with patch_instance(fields):
        scripted_model = torch.jit.script(model)
        return scripted_model


@contextmanager
def patch_instance(fields):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as f:
        try:
            cls_name, s = _gen_module(fields)
            f.write(s)
            f.flush()
            f.close()

            module = _import(f.name)
            new_instance = getattr(module, cls_name)
            _ = torch.jit.script(new_instance)

            Instances.__torch_script_class__ = True
            Instances._jit_override_qualname = torch._jit_internal._qualified_name(new_instance)
            yield new_instance
        finally:
            sys.modules.pop(module.__name__)
            del Instances.__torch_script_class__
            del Instances._jit_override_qualname


# todo: find a more automatic way to enable the import of other classes
def _gen_imports():

    imports_str = """
import torch
from torch import Tensor
import typing
from typing import *

from detectron2.structures import Boxes

"""
    return imports_str


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


def _gen_module(fields):
    s = ""
    s += _gen_imports()
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
