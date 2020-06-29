FROM nvidia/cuda:10.1-cudnn7-devel
# This dockerfile only aims to provide an environment for unittest on CircleCI

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo ninja-build libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
RUN pip install tensorboard opencv-python-headless
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} -f https://download.pytorch.org/whl/cu101/torch_stable.html
