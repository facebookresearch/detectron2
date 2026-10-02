# Using Ubuntu 20.04 is creating an overall better impact in the download and installation
# The previous Dockerfile with 18.04 ubuntu was causing multiple errors
# This script creates a Docker image that can directly be run to get the output
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Setting non-interactive terminal to avoid input during process
ENV DEBIAN_FRONTEND=noninteractive

# Installing necessary deps
# A virtual environment gives better support overall
RUN apt-get update && apt-get install -y python3-opencv python3-venv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

# install dependencies

# Create a virtual env in the /tmp directory
# Change path if you want
RUN python3 -m venv /tmp/detectron_env

# Set the Virtual ENV and Cuda ENV paths and add them to the base env PATHs
# Add LD Library Path as well
ENV VIRTUAL_ENV=/tmp/detectron_env
ENV CUDA_HOME=/usr/local/cuda-11.1
ENV PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64"

# Define all necessary CUDA architectures
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install all necessary packages using pip in our virtual environment
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install opencv-python
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# Clone detectron repo install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

# The demo file does not run for some reason so omit that
# If anyone does make changes to demo.py, make sure to let me know at aarambh.patil.8703@gmail.com

# run detectron2 under user "appuser":
# RUN wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# RUN python3 demo/demo.py  \
# 	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
# 	--input input.jpg --output outputs/ \
# 	--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
