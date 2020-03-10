# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import importlib
import numpy as np
import os
import re
import subprocess
import sys
from collections import defaultdict
import PIL
import torch
import torchvision
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


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            sm = []
            for line in output:
                line = re.findall(r"\.sm_[0-9]*\.", line)[0]
                sm.append(line.strip("."))
            sm = sorted(set(sm))
            return ", ".join(sm)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    has_cuda = torch.cuda.is_available()
    # NOTE: the use of CUDA_HOME requires the CUDA build deps, though in
    # theory detectron2 should be made runnable with only the CUDA runtime
    from torch.utils.cpp_extension import CUDA_HOME

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import detectron2  # noqa

        data.append(
            ("detectron2", detectron2.__version__ + " @" + os.path.dirname(detectron2.__file__))
        )
    except ImportError:
        data.append(("detectron2", "failed to import"))
    else:
        try:
            from detectron2 import _C
        except ImportError:
            data.append(("detectron2._C", "failed to import"))
        else:
            data.append(("detectron2 compiler", _C.get_compiler_version()))
            data.append(("detectron2 CUDA compiler", _C.get_cuda_version()))
            if has_cuda:
                data.append(
                    ("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, _C.__file__))
                )

    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__ + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("CUDA available", has_cuda))
    if has_cuda:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        from torch.utils.cpp_extension import CUDA_HOME

        data.append(("CUDA_HOME", str(CUDA_HOME)))

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = subprocess.check_output("'{}' -V | tail -n1".format(nvcc), shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            data.append(("NVCC", nvcc))

        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list:
            data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "failed to find"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    try:
        import detectron2  # noqa
    except ImportError:
        print(collect_env_info())
    else:
        from detectron2.utils.collect_env import collect_env_info

        print(collect_env_info())
