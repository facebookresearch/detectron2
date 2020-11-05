#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "detectron2", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "detectron2", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )

    if is_rocm_pytorch:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/detectron2/layers/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        # Current version of hipify function in pytorch creates an intermediate directory
        # named "hip" at the same level of the path hierarchy if a "cuda" directory exists,
        # or modifying the hierarchy, if it doesn't. Once pytorch supports
        # "same directory" hipification (https://github.com/pytorch/pytorch/pull/40523),
        # the source_cuda will be set similarly in both cuda and hip paths, and the explicit
        # header file copy (below) will not be needed.
        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        shutil.copy(
            "detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h",
            "detectron2/layers/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h",
        )
        shutil.copy(
            "detectron2/layers/csrc/deformable/deform_conv.h",
            "detectron2/layers/csrc/deformable/hip/deform_conv.h",
        )

    else:
        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

    sources = [main_source] + sources
    sources = [
        s
        for s in sources
        if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
    ]

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    detectron2/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths


# For projects that are relative small and provide features that are very close
# to detectron2's core functionalities, we install them under detectron2.projects
PROJECTS = {
    "detectron2.projects.point_rend": "projects/PointRend/point_rend",
    "detectron2.projects.deeplab": "projects/DeepLab/deeplab",
    "detectron2.projects.panoptic_deeplab": "projects/Panoptic-DeepLab/panoptic_deeplab",
}

setup(
    name="detectron2",
    version=get_version(),
    author="FAIR",
    url="https://github.com/facebookresearch/detectron2",
    description="Detectron2 is FAIR's next-generation research "
    "platform for object detection and segmentation.",
    packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    package_dir=PROJECTS,
    package_data={"detectron2.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.6",
    install_requires=[
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        "termcolor>=1.1",
        "Pillow>=7.1",  # or use pillow-simd for better performance
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.2",
        "pycocotools>=2.0.2",  # corresponds to https://github.com/ppwwyyxx/cocoapi
        "future",  # used by caffe2
        "pydot",  # used to save caffe2 SVGs
    ],
    extras_require={
        "all": [
            "shapely",
            "psutil",
            "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
        ],
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "black==20.8b1",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
