# Deployment

Models written in Python need to go through an export process to become a deployable artifact.
A few basic concepts about this process:

__"Export method"__ is how a Python model is fully serialized to a deployable format.
We support the following export methods:

* `tracing`: see [pytorch documentation](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) to learn about it\
This method produces a standardized ONNX graph that can be used by any ONNX runtime, such as \
[ONNX Runtime](https://onnxruntime.ai/)
* `scripting`: see [pytorch documentation](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) to learn about it\
This method produces a standardized ONNX graph that can be used by any ONNX runtime, such as \
[ONNX Runtime](https://onnxruntime.ai/)
* `caffe2_tracing`: replace parts of the model by caffe2 operators, then use tracing.
This method produces a Caffe2-specific ONNX graph that can be used only by the [Caffe2](https://caffe2.ai/).
Caffe2 support has been ***DEPRECATED*** by PyTorch, therefore, we don't plan to support it, but contributions are welcome.


__"Format"__ is how a serialized model is stored in a file, e.g.
TorchScript, ONNX format or Caffe2 protobuf. We don't plan to work on additional support for other formats, but contributions are welcome.

__"Runtime"__ is an engine that loads a serialized model and executes it,
e.g., PyTorch, ONNX Runtime, TensorFlow, TensorRT, (DEPRECATED) Caffe2, etc.
A runtime is often tied to a specific format
(e.g. PyTorch needs TorchScript format, ONNX Runtime requires ONNX format, (DEPRECATED) Caffe2 needs protobuf format).
We currently support the following combination and each has some limitations:

```eval_rst
+----------------------------+-------------+-------------+-------------------+-----------------------------+
|       Export Method        |   tracing   |  scripting  | tracing/scripting | caffe2_tracing (DEPRECATED) |
+============================+=============+=============+===================+=============================+
| **Formats**                | TorchScript | TorchScript |       ONNX        | Caffe2, TorchScript, ONNX   |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| **Runtime**                | PyTorch     | PyTorch     |    ONNX Runtime   | Caffe2, PyTorch             |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| C++/Python inference       | ✅           | ✅           | ✅                 | ✅                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| Dynamic resolution         | ✅           | ✅           | ✅                 | ✅                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| Batch size requirement     | Constant    | Dynamic     | Dynamic           | Batch inference unsupported |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| Extra runtime deps         | torchvision | torchvision | torchvision       | Caffe2 ops (usually already |
|                            |             |             |                   |                             |
|                            |             |             |                   | included in PyTorch)        |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| Faster/Mask/Keypoint R-CNN | ✅           | ✅           | ✅                 | ✅                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| RetinaNet                  | ✅           | ✅           | ✅                 | ✅                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| PointRend R-CNN            | ✅           | ❌           | ✅                 | ❌                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| Cascade R-CNN              | ✅           | ❌           | ✅                 | ❌                           |
+----------------------------+-------------+-------------+-------------------+-----------------------------+
| PanopticFPN                | ✅           | N/A         | ✅                 | N/A                         |
+----------------------------+-------------+-------------+-------------------+-----------------------------+

```

## Deployment with Tracing or Scripting

[//]: # (Make distinction between TorchScript and ONNX when ONNX Runtime is on CI)
Models can be exported to TorchScript or ONNX format, by either
[tracing or scripting](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
[ONNX format](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) can be used
with both export methods.
The output model file can be loaded without `detectron2` dependency in either Python or C++.
The exported model often requires `torchvision` (or its C++ library) dependency for some custom ops.

This feature requires PyTorch ≥ 1.8.

### Coverage

Most official models under the meta architectures `GeneralizedRCNN` and `RetinaNet`
are supported in both tracing and scripting mode.
`Cascade R-CNN`, `PointRend` and `PanopticFPN` are currently supported in tracing.
Users' custom extensions are supported if they are also scriptable or traceable.

For models exported with tracing, dynamic input resolution is allowed, but batch size
(number of input images) must be fixed.
Scripting can support dynamic batch size.

### Usage

The main export APIs for tracing and scripting are [TracingAdapter](../modules/export.html#detectron2.export.TracingAdapter)
and [scripting_with_instances](../modules/export.html#detectron2.export.scripting_with_instances).
Their usage is currently demonstrated in [test_export_torchscript.py](../../tests/test_export_torchscript.py)
(see `TestScripting` and `TestTracing`)
as well as the [deployment example](../../tools/deploy).
Please check that these examples can run, and then modify for your use cases.
The usage now requires some user effort and necessary knowledge for each model to workaround the limitation of scripting and tracing.
In the future we plan to wrap these under simpler APIs to lower the bar to use them.

A quick example for using `tools/deploy/export_models.py` is:

## (DEPRECATED) Deployment with Caffe2-tracing

We provide [Caffe2Tracer](../modules/export.html#detectron2.export.Caffe2Tracer)
that performs the export logic.
It replaces parts of the model with Caffe2 operators (which are not part of ONNX standard),
and then export the model into Caffe2, TorchScript or ONNX format.

The converted model is able to run in either Python or C++ without `detectron2`/`torchvision` dependency, on CPU or GPUs.
It has a runtime optimized for CPU & mobile inference, but not optimized for GPU inference.

Because Caffe2 has been deprecated, in order to run the model, [PyTorch](http://pytorch.org/) has to be
[compiled from source](https://github.com/pytorch/pytorch#from-source) with the build flag `BUILD_CAFFE2=1`.
There are no pre-built PyTorch packages with Caffe2 publicly available.
This feature requires ONNX ≥ 1.6.

### Coverage

Most official models under these 3 common meta architectures: `GeneralizedRCNN`, `RetinaNet`, `PanopticFPN`
are supported. Cascade R-CNN is not supported. Batch inference is not supported.

Users' custom extensions under these architectures (added through registration) are supported
as long as they do not contain control flow or operators not available in Caffe2 (e.g. deformable convolution).
For example, custom backbones and heads are often supported out of the box.

### Usage

The APIs are listed at [the API documentation](../modules/export).
We provide [export_model.py](../../tools/deploy/) as an example that uses
these APIs to convert a standard model. For custom models/datasets, you can add them to this script.

### Use the model in C++/Python

The model can be loaded in C++ and deployed with
either [ONNX Runtime](https://onnxruntime.ai), or [Pytorch](https://pytorch.org) for models exported using
tracing or scripting. The deprecated Caffe2 compiled from source can be used to deploy models exported
using tracing, scripting or caffe2_tracing.
Refer to [C++ examples](../../tools/deploy/) to checkout usage examples for Mask R-CNN.

Note that the converted models do not contain post-processing operations that
transform raw layer outputs into formatted predictions.
For example, the C++ examples only produce raw outputs (28x28 masks) from the final
layers that are not post-processed, because in actual deployment, an application often needs
its custom lightweight post-processing, so this step is left for users.

#### (DEPRECATED) Caffe2 notes

For Models exported with `caffe2_tracing` method, note that:

* The resulting model takes a special input format described in
[documentation](../modules/export.html#detectron2.export.Caffe2Tracer).
This was taken care of in the C++ example.

* To run the model, a prebuilt PyTorch package (e.g. from `pip install torch`) cannot be used.
Instead, you have to compile PyTorch from source with Caffe2 lib support.

* To help using the Caffe2-format model in python,
we provide a python wrapper around the converted model, in the
[Caffe2Model.\_\_call\_\_](../modules/export.html#detectron2.export.Caffe2Model.__call__) method.
This method has an interface that's identical to the [pytorch versions of models](./models.md),
and it internally applies pre/post-processing code to match the formats.
This wrapper can serve as a reference for how to use Caffe2's python API,
or for how to implement pre/post-processing in actual deployment.

## Conversion to TensorFlow

[tensorpack Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/convert_d2)
provides scripts to convert a few standard `detectron2` R-CNN models to TensorFlow's pb format.
It works by translating configs and weights, therefore only support a few models.
