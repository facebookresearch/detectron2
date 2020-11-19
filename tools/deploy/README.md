See [deployment tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for some high-level background about deployment.

This directory contains:

1. A script `caffe2_converter.py` that converts a detectron2 model using caffe2-style tracing,
   into caffe2, onnx, or torchscript format.

2. Two C++ examples that run inference with Mask R-CNN model in caffe2/torchscript format.

## Build
The C++ examples need to be built with:
* PyTorch with caffe2 inside
* gflags, glog, opencv
* protobuf library that match the version used by PyTorch (version defined in `include/caffe2/proto/caffe2.pb.h` of your PyTorch installation)
* MKL headers if caffe2 is built with MKL

As a reference, the following steps can build the C++ example inside [official detectron2 docker](../../docker/).
```
# install dependencies
sudo apt update && sudo apt install libgflags-dev libgoogle-glog-dev libopencv-dev
pip install mkl-include

# install the correct version of protobuf:
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-cpp-3.11.4.tar.gz && tar xf protobuf-cpp-3.11.4.tar.gz
cd protobuf-3.11.4
export CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=$(python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
./configure --prefix=$HOME/.local && make && make install
export CPATH=$HOME/.local/include
export LIBRARY_PATH=$HOME/.local/lib
export LD_LIBRARY_PATH=$HOME/.local/lib

# build the program:
export CMAKE_PREFIX_PATH=$HOME/.local/lib/python3.6/site-packages/torch/
mkdir build && cd build
cmake -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST .. && make
```

## Use
First, convert a model:
```
# caffe2 format:
./caffe2_converter.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--output ./output --format caffe2 --run-eval \
MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
MODEL.DEVICE cpu

# torchscript format:
./caffe2_converter.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--output ./output --format torchscript \
MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
MODEL.DEVICE cpu
```

Then, run the C++ applications:
```
./caffe2_mask_rcnn --predict_net=output/model.pb --init_net=output/model_init.pb --input=input.jpg

./torchscript_traced_mask_rcnn output/model.ts input.jpg
```
