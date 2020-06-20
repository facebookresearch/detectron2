#!/bin/bash -e
# This script is used for fast install detectron2 requirements.
# Note: only support cuda10.1.

# For dev
pip install flake8
pip install pre-commit

pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install cython pyyaml==5.1
pip install opencv-python

# Cocoapi, if too slow, you can try use below method to install cocoapi
# pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone --branch master https://github.com/cocodataset/cocoapi.git --depth 1
cd cocoapi/PythonAPI
python setup.py install
cd -
rm -rf cocoapi

python -m pip install -e .
