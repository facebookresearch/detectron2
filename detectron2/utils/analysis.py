# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-

import logging
import typing
import torch
from fvcore.nn import activation_count, flop_count, parameter_count, parameter_count_table
from torch import nn

from .logger import log_first_n

__all__ = [
    "activation_count_operators",
    "flop_count_operators",
    "parameter_count_table",
    "parameter_count",
]

FLOPS_MODE = "flops"
ACTIVATIONS_MODE = "activations"


def flop_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of fvcore.nn.flop_count, that supports standard detection models
    in detectron2.

    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=FLOPS_MODE, **kwargs)


def activation_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level activations counting using jit.
    This is a wrapper of fvcore.nn.activation_count, that supports standard detection models
    in detectron2.

    Note:
        The function runs the input through the model to compute activations.
        The activations of a detection model is often input-dependent, for example,
        the activations of box & mask head depends on the number of proposals &
        the number of detected objects.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=ACTIVATIONS_MODE, **kwargs)


def _wrapper_count_operators(
    model: nn.Module, inputs: list, mode: str, **kwargs
) -> typing.DefaultDict[str, float]:

    assert len(inputs) == 1, "Please use batch size=1"
    tensor_input = inputs[0]["image"]

    class WrapModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            if isinstance(
                model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
            ):
                self.model = model.module
            else:
                self.model = model

        def forward(self, image):
            # jit requires the input/output to be Tensors
            inputs = [{"image": image}]
            outputs = self.model.forward(inputs)[0]
            if isinstance(outputs, dict) and "instances" in outputs:
                # Only the subgraph that computes the returned tensor will be
                # counted. So we return everything we found in Instances.
                inst = outputs["instances"]
                ret = [inst.pred_boxes.tensor]
                inst.remove("pred_boxes")
                for k, v in inst.get_fields().items():
                    if isinstance(v, torch.Tensor):
                        ret.append(v)
                    else:
                        log_first_n(
                            logging.WARN,
                            f"Field '{k}' in output instances is not included"
                            " in flops/activations count.",
                            n=10,
                        )
                return tuple(ret)
            raise NotImplementedError("Count for segmentation models is not supported yet.")

    old_train = model.training
    with torch.no_grad():
        if mode == FLOPS_MODE:
            ret = flop_count(WrapModel(model).train(False), (tensor_input,), **kwargs)
        elif mode == ACTIVATIONS_MODE:
            ret = activation_count(WrapModel(model).train(False), (tensor_input,), **kwargs)
        else:
            raise NotImplementedError("Count for mode {} is not supported yet.".format(mode))
    # compatible with change in fvcore
    if isinstance(ret, tuple):
        ret = ret[0]
    model.train(old_train)
    return ret
