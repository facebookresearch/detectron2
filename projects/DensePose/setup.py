import re
from pathlib import Path
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


def get_detectron2_current_version():
    """Version is not available for import through Python since it is
    above the top level of the package. Instead, we parse it from the
    file with a regex."""
    # Get version info from detectron2 __init__.py
    version_source = (Path(__file__).parents[2] / "detectron2" / "__init__.py").read_text()
    version_number = re.findall(r'__version__ = "([0-9\.]+)"', version_source)[0]
    return version_number


setup(
    name="detectron2-densepose",
    author="FAIR",
    version=get_detectron2_current_version(),
    url="https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "av>=8.0.3",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
        "opencv-python-headless>=4.5.3.56",
        "scipy>=1.5.4",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
)
