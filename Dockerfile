FROM nvidia/cuda:10.1-cudnn7-devel
# To use this Dockerfile:
# 1. `nvidia-docker build -t detectron2:v0 .`
# 2. `nvidia-docker run -it --name detectron2 detectron2:v0`


ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision cython \
	'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e /detectron2_repo

WORKDIR /detectron2_repo

# run it, for example:
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

