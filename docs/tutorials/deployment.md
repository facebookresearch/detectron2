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

The conversion APIs are documented at [the API documentation](../modules/export.html).
We provide a tool, `tools/caffe2_converter.py` as an example that uses
these APIs to convert a standard model.

To convert an official Mask R-CNN trained on COCO, first
[prepare the COCO dataset](../../datasets/), then pick the model from [Model Zoo](../../MODEL_ZOO.md), and run:
```
cd tools/ && ./caffe2_converter.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	--output ./caffe2_model --run-eval \
	MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
	MODEL.DEVICE cpu
```

Note that:
1. The conversion needs valid sample inputs & weights to trace the model. That's why the script requires the dataset.
	 You can modify the script to obtain sample inputs in other ways.
2. GPU conversion is supported only with Pytorch's master. So we use `MODEL.DEVICE cpu`.
3. With the `--run-eval` flag, it will evaluate the converted models to verify its accuracy.
   The accuracy is typically slightly different (within 0.1 AP) from PyTorch due to
	 numerical precisions between different implementations.
	 It's recommended to always verify the accuracy in case your custom model is not supported by the
	 conversion.

The converted model is available at the specified `caffe2_model/` directory. Two files `model.pb`
and `model_init.pb` that contain network structure and network parameters are necessary for deployment.
These files can then be loaded in C++ or Python using Caffe2's APIs.

The script generates `model.svg` file which contains a visualization of the network.
You can also load `model.pb` to tools such as [netron](https://github.com/lutzroeder/netron) to visualize it.

### Inputs & Outputs

All converted models (the .pb file) take two input tensors:
"data" which is an NCHW image, and "im_info" which is a Nx3 tensor of (height, width, unused legacy parameter) for
each image (the shape of "data" might be larger than that in "im_info" due to padding).

The converted models do not contain post-processing operations that
transform raw layer outputs into formatted predictions.
The models only produce raw outputs from the final
layers that are not post-processed, because in actual deployment, an application often needs
its custom lightweight post-processing (e.g. full-image masks for every detected object is often not necessary).

Due to different inputs & outputs formats,
we provide a wrapper around the converted model, in the [Caffe2Model.__call__](../modules/export.html#detectron2.export.Caffe2Model.__call__) method.
It has an interface that's identical to the [format of pytorch versions of models](models.html),
and it internally applies pre/post-processing code to match the formats.
They can serve as a reference for pre/post-processing in actual deployment.