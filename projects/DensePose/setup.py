from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
except ImportError:
    raise Exception(
        """
You must install PyTorch prior to installing DensePose:
pip install torch

For more information:
    https://pytorch.org/get-started/locally/
    """
    )

setup(
    name="DensePose",
    author="FAIR",
    url="https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "av >= 8.0.3",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
        "opencv-python-headless >= 4.5.3.56",
        "scipy >= 1.5.4",
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
    ],
)
