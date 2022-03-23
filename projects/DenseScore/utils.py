# Copyright (c) Facebook, Inc. and its affiliates.

from torch import nn


def initialize_module_params(module: nn.Module):
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif len(param.shape) == 1:
                nn.init.constant_(param, 1)
