# Deployment

## Caffe2 Deployment
We currently support converting a detectron2 model to Caffe2 format through ONNX.
The converted Caffe2 model is able to run without detectron2 dependency in either Python or C++.
It has a runtime optimized for CPU & mobile inference, but not for GPU inference.

Caffe2 conversion requires PyTorch ≥ 1.4 and ONNX ≥ 1.6.

### Coverage

It supports 3 most common meta architectures: `GeneralizedRCNN`, `RetinaNet`, `PanopticFPN`,
and most official models under these 3 meta architectures.

Users' custom extensions under these architectures (added through registration) are supported
as long as they do not contain control flow or operators not available in Caffe2 (e.g. deformable convolution).
For example, custom backbones and heads are often supported out of the box.

### Usage

The conversion APIs are documented at [the API documentation](../modules/export).
We provide a tool, `caffe2_converter.py` as an example that uses
these APIs to convert a standard model.

To convert an official Mask R-CNN trained on COCO, first
[prepare the COCO dataset](../../datasets/), then pick the model from [Model Zoo](../../MODEL_ZOO.md), and run:
```
cd tools/deploy/ && ./caffe2_converter.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	--output ./caffe2_model --run-eval \
	MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
	MODEL.DEVICE cpu
```

Note that:
1. The conversion needs valid sample inputs & weights to trace the model. That's why the script requires the dataset.
	 You can modify the script to obtain sample inputs in other ways.
2. With the `--run-eval` flag, it will evaluate the converted models to verify its accuracy.
   The accuracy is typically slightly different (within 0.1 AP) from PyTorch due to
	 numerical precisions between different implementations.
	 It's recommended to always verify the accuracy in case your custom model is not supported by the
	 conversion.

The converted model is available at the specified `caffe2_model/` directory. Two files `model.pb`
and `model_init.pb` that contain network structure and network parameters are necessary for deployment.
These files can then be loaded in C++ or Python using Caffe2's APIs.

The script generates `model.svg` file which contains a visualization of the network.
You can also load `model.pb` to tools such as [netron](https://github.com/lutzroeder/netron) to visualize it.

### Use the model in C++/Python

The model can be loaded in C++. An example [caffe2_mask_rcnn.cpp](../../tools/deploy/) is given,
which performs CPU/GPU inference using `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x`.

The C++ example needs to be built with:
* PyTorch with caffe2 inside
* gflags, glog, opencv
* protobuf headers that match the version of your caffe2
* MKL headers if caffe2 is built with MKL

The following can compile the example inside [official detectron2 docker](../../docker/):
```
sudo apt update && sudo apt install libgflags-dev libgoogle-glog-dev libopencv-dev
pip install mkl-include
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.tar.gz
tar xf protobuf-cpp-3.6.1.tar.gz
export CPATH=$(readlink -f ./protobuf-3.6.1/src/):$HOME/.local/include
export CMAKE_PREFIX_PATH=$HOME/.local/lib/python3.6/site-packages/torch/
mkdir build && cd build
cmake -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST .. && make

# To run:
./caffe2_mask_rcnn --predict_net=./model.pb --init_net=./model_init.pb --input=input.jpg
```

Note that:

* All converted models (the .pb files) take two input tensors:
  "data" is an NCHW image, and "im_info" is an Nx3 tensor consisting of (height, width, 1.0) for
  each image (the shape of "data" might be larger than that in "im_info" due to padding).

* The converted models do not contain post-processing operations that
  transform raw layer outputs into formatted predictions.
  The example only produces raw outputs (28x28 masks) from the final
  layers that are not post-processed, because in actual deployment, an application often needs
  its custom lightweight post-processing (e.g. full-image masks for every detected object is often not necessary).

We also provide a python wrapper around the converted model, in the
[Caffe2Model.\_\_call\_\_](../modules/export.html#detectron2.export.Caffe2Model.__call__) method.
This method has an interface that's identical to the [pytorch versions of models](./models.md),
and it internally applies pre/post-processing code to match the formats.
They can serve as a reference for pre/post-processing in actual deployment.
