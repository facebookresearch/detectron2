FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo && \
  rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV http_proxy="192.168.1.11:8118"
ENV	https_proxy="192.168.1.11:8118"
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
# todo: unset proxy because proxy will forward request to tsinghua pypi host as well
# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
ENV PIP_INSTALL="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple" 
RUN $PIP_INSTALL --user torch torchvision tensorboard cython
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# RUN $PIP_INSTALL --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/model_cache"
WORKDIR /home/appuser/detectron2_repo
# these copied from setup.py, could be updates there
RUN pip install --user "termcolor>=1.1" \
        "Pillow==6.2.2" \
        "yacs>=0.1.6" \
        "tabulate" \
        "cloudpickle" \
        "matplotlib" \
        "tqdm>4.29.0" \
        "tensorboard" \
        "fvcore" \
        "future" \
        "pydot" \
		"pandas"
# run it, for example:
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
