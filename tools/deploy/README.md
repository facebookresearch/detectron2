See [deployment tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for some high-level background about deployment.

This directory contains the following examples:

1. An example script `export_model.py` (previously called `caffe2_converter.py`)
   that exports a detectron2 model for deployment using different methods and formats.

2. A few C++ examples that run inference with Mask R-CNN model in Caffe2/TorchScript format.

## Build
All C++ examples depend on libtorch and OpenCV. Some require more dependencies:

* Running caffe2-format models requires:
  * libtorch built with caffe2 inside
  * gflags, glog
  * protobuf library that matches the version used by PyTorch (version defined in `include/caffe2/proto/caffe2.pb.h` of your PyTorch installation)
  * MKL headers if caffe2 is built with MKL
* Running TorchScript-format models produced by `--export-method=caffe2_tracing` requires no other dependencies.
* Running TorchScript-format models produced by `--export-method=tracing` requires libtorchvision (C++ library of torchvision).

We build all examples with one `CMakeLists.txt` that requires all the above dependencies.
Adjust it if you only need one example.
As a reference,
we provide a [Dockerfile](../../docker/deploy.Dockerfile) that
installs all the above dependencies and builds the C++ examples.

## Use

We show a few example commands to export and execute a Mask R-CNN model in C++.

* `export-method=caffe2_tracing, format=caffe2`:
```
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method caffe2_tracing --format caffe2 \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    MODEL.DEVICE cpu

./build/caffe2_mask_rcnn --predict_net=output/model.pb --init_net=output/model_init.pb --input=input.jpg
```

* `export-method=caffe2_tracing, format=torchscript`:

```
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method caffe2_tracing --format torchscript \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    MODEL.DEVICE cpu

./build/torchscript_traced_mask_rcnn output/model.ts input.jpg caffe2_tracing
```

* `export-method=tracing, format=torchscript`:

```
# this example also tries GPU instead of CPU
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method tracing --format torchscript \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    MODEL.DEVICE cuda

./build/torchscript_traced_mask_rcnn output/model.ts input.jpg tracing
```

## Notes:

1. Tracing/Caffe2-tracing requires valid weights & sample inputs.
   Therefore the above commands require pre-trained models and [COCO dataset](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html).
   You can modify the script to obtain sample inputs in other ways instead of from COCO.

2. `--run-eval` flag can be used under certain modes
   (caffe2_tracing with caffe2 format, or tracing with torchscript format)
   to evaluate the exported model using the dataset in the config.
   It's recommended to always verify the accuracy in case the conversion is not successful.
   Evaluation can be slow if model is exported to CPU or dataset is too large ("coco_2017_val_100" is a small subset of COCO useful for evaluation).
   Caffe2 accuracy may be slightly different (within 0.1 AP) from original model due to numerical precisions between different runtime.
