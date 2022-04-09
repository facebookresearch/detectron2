See [deployment tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for some high-level background about deployment.

This directory contains the following examples:

1. An example script `export_model.py`
   that exports a detectron2 model for deployment using different methods and formats.

2. A C++ example that runs inference with Mask R-CNN model in TorchScript format.

## Build
Deployment depends on libtorch and OpenCV. Some require more dependencies:

* Running TorchScript-format models produced by `--export-method=caffe2_tracing` requires libtorch
  to be built with caffe2 enabled.
* Running TorchScript-format models produced by `--export-method=tracing/scripting` requires libtorchvision (C++ library of torchvision).

All methods are supported in one C++ file that requires all the above dependencies.
Adjust it and remove code you don't need.
As a reference, we provide a [Dockerfile](../../docker/deploy.Dockerfile) that installs all the above dependencies and builds the C++ example.

## Use

We show a few example commands to export and execute a Mask R-CNN model in C++.

* `export-method=tracing, format=torchscript`:
```
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method tracing --format torchscript \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
    MODEL.DEVICE cuda

./build/torchscript_mask_rcnn output/model.ts input.jpg tracing
```

* `export-method=scripting, format=torchscript`:
```
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method scripting --format torchscript \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \

./build/torchscript_mask_rcnn output/model.ts input.jpg scripting
```

* `export-method=caffe2_tracing, format=torchscript`:

```
./export_model.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --output ./output --export-method caffe2_tracing --format torchscript \
    MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \

./build/torchscript_mask_rcnn output/model.ts input.jpg caffe2_tracing
```


## Notes:

1. Tracing/Caffe2-tracing requires valid weights & sample inputs.
   Therefore the above commands require pre-trained models and [COCO dataset](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html).
   You can modify the script to obtain sample inputs in other ways instead of from COCO.

2. `--run-eval` is implemented only for tracing mode
   to evaluate the exported model using the dataset in the config.
   It's recommended to always verify the accuracy in case the conversion is not successful.
   Evaluation can be slow if model is exported to CPU or dataset is too large ("coco_2017_val_100" is a small subset of COCO useful for evaluation).
   `caffe2_tracing` accuracy may be slightly different (within 0.1 AP) from original model due to numerical precisions between different runtime.
