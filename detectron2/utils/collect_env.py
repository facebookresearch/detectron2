# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
from collections import defaultdict
import PIL
import torch
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_env_info():
    data = []
    data.append(("Python", sys.version.replace("\n", "")))
    try:
        from detectron2 import _C
    except ImportError:
        pass
    else:
        data.append(("Detectron2 Compiler", _C.get_compiler_version()))

    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch Debug Build", torch.version.debug))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
    data.append(("Pillow", PIL.__version__))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    print(collect_env_info())
