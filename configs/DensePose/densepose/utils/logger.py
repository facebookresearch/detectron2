# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
import logging


def verbosity_to_level(verbosity) -> int:
    if verbosity is not None:
        if verbosity == 0:
            return logging.WARNING
        elif verbosity == 1:
            return logging.INFO
        elif verbosity >= 2:
            return logging.DEBUG
    return logging.WARNING
