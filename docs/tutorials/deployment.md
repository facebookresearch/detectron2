# Deployment

## TorchScript Deployment

Models can be exported to TorchScript format, by either
[tracing or scripting](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
The output model file can be loaded without detectron2 dependency in either Python or C++.
The exported model likely require torchvision (or its C library) dependency for some custom ops.

This feature requires PyTorch ≥ 1.8 (or latest on github before 1.8 is released).

### Coverage
Most official models under the meta architectures `GeneralizedRCNN` and `RetinaNet`
are supported in both tracing and scripting mode. Cascade R-CNN is not supported.
Users' custom extensions are supported if they are also scriptable or traceable.

For models exported with tracing, dynamic input resolution is allowed, but batch size
(number of input images) must be fixed.
Scripting can support dynamic batch size.

### Usage

The usage is currently demonstrated in [test_export_torchscript.py](https://github.com/facebookresearch/detectron2/blob/master/tests/test_export_torchscript.py)
(see `TestScripting` and `TestTracing`).
It shows that the current usage requires some user effort (and necessary knowledge) for each model to workaround the limitation of scripting and tracing.
In the future we plan to wrap these under simpler APIs, and provide a complete export and deployment example to lower the bar to use them.

## Caffe2 Deployment
We support converting a detectron2 model to Caffe2 format through ONNX.
The converted Caffe2 model is able to run without detectron2 dependency in either Python or C++.
It has a runtime optimized for CPU & mobile inference, but not for GPU inference.

Caffe2 conversion requires ONNX ≥ 1.6.

### Coverage

Most official models under these 3 common meta architectures: `GeneralizedRCNN`, `RetinaNet`, `PanopticFPN`
are supported. Cascade R-CNN is not supported. Batch inference is not supported.

Users' custom extensions under these architectures (added through registration) are supported
as long as they do not contain control flow or operators not available in Caffe2 (e.g. deformable convolution).
For example, custom backbones and heads are often supported out of the box.

### Usage

The conversion APIs are documented at [the API documentation](../modules/export).
We provide a tool, `caffe2_converter.py` as an example that uses
these APIs to convert a standard model.

To convert an official Mask R-CNN trained on COCO, first
[prepare the COCO dataset](builtin_datasets.md), then pick the model from [Model Zoo](../../MODEL_ZOO.md), and run:
```
cd tools/deploy/ && ./caffe2_converter.py --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --output ./caffe2_model --run-eval \
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
  MODEL.DEVICE cpu
```

Note that:
1. The conversion needs valid weights & sample inputs to trace the model. That's why the script requires the dataset.
   You can modify the script to obtain sample inputs in other ways.
2. With the `--run-eval` flag, it will evaluate the converted models to verify its accuracy.
   The accuracy is typically slightly different (within 0.1 AP) from PyTorch due to
   numerical precisions between different implementations.
   It's recommended to always verify the accuracy in case the conversion is not successful.

The converted model is available at the specified `caffe2_model/` directory. Two files `model.pb`
and `model_init.pb` that contain network structure and network parameters are necessary for deployment.
These files can then be loaded in C++ or Python using Caffe2's APIs.

The script generates `model.svg` file which contains a visualization of the network.
You can also load `model.pb` to tools such as [netron](https://github.com/lutzroeder/netron) to visualize it.

### Use the model in C++/Python

The model can be loaded in C++. [C++ examples](../../tools/deploy/) for Mask R-CNN
are given as references. Note that:

* All converted models (the .pb files) take two input tensors:
  "data" is an NCHW image, and "im_info" is an Nx3 tensor consisting of (height, width, 1.0) for
  each image (the shape of "data" might be larger than that in "im_info" due to padding).
  This was taken care of in the C++ example.

* The converted models do not contain post-processing operations that
  transform raw layer outputs into formatted predictions.
  For example, the C++ examples only produce raw outputs (28x28 masks) from the final
  layers that are not post-processed, because in actual deployment, an application often needs
  its custom lightweight post-processing, so this step is left for users.

To use the converted model in python,
we provide a python wrapper around the converted model, in the
[Caffe2Model.\_\_call\_\_](../modules/export.html#detectron2.export.Caffe2Model.__call__) method.
This method has an interface that's identical to the [pytorch versions of models](./models.md),
and it internally applies pre/post-processing code to match the formats.
This wrapper can serve as a reference for how to use caffe2's python API,
or for how to implement pre/post-processing in actual deployment.

## Conversion to TensorFlow
[tensorpack Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/convert_d2)
provides scripts to convert a few standard detectron2 R-CNN models to TensorFlow's pb format.
It works by translating configs and weights, therefore only support a few models.
