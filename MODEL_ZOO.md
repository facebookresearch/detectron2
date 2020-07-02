# Detectron2 Model Zoo and Baselines

## Introduction

This file documents a large collection of baselines trained
with detectron2 in Sep-Oct, 2019.
All numbers were obtained on [Big Basin](https://engineering.fb.com/data-center-engineering/introducing-big-basin-our-next-generation-ai-hardware/)
servers with 8 NVIDIA V100 GPUs & NVLink. The software in use were PyTorch 1.3, CUDA 9.2, cuDNN 7.4.2 or 7.6.3.
You can access these models from code using [detectron2.model_zoo](https://detectron2.readthedocs.io/modules/model_zoo.html) APIs.

In addition to these official baseline models, you can find more models in [projects/](projects/).

#### How to Read the Tables
* The "Name" column contains a link to the config file. Running `tools/train_net.py` with this config file
	and 8 GPUs will reproduce the model.
* Training speed is averaged across the entire training.
	We keep updating the speed with latest version of detectron2/pytorch/etc.,
	so they might be different from the `metrics` file.
	Training speed for multi-machine jobs is not provided.
* Inference speed is measured by `tools/train_net.py --eval-only`, or [inference_on_dataset()](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.inference_on_dataset),
  with batch size 1 in detectron2 directly.
	Measuring it with custom code may introduce other overhead.
  Actual deployment in production should in general be faster than the given inference
  speed due to more optimizations.
* The *model id* column is provided for ease of reference.
  To check downloaded file integrity, any model on this page contains its md5 prefix in its file name.
* Training curves and other statistics can be found in `metrics` for each model.

#### Common Settings for COCO Models
* All COCO models were trained on `train2017` and evaluated on `val2017`.
* The default settings are __not directly comparable__ with Detectron's standard settings.
  For example, our default training data augmentation uses scale jittering in addition to horizontal flipping.

  To make fair comparisons with Detectron's settings, see
  [Detectron1-Comparisons](configs/Detectron1-Comparisons/) for accuracy comparison,
  and [benchmarks](https://detectron2.readthedocs.io/notes/benchmarks.html)
  for speed comparison.
* For Faster/Mask R-CNN, we provide baselines based on __3 different backbone combinations__:
  * __FPN__: Use a ResNet+FPN backbone with standard conv and FC heads for mask and box prediction,
    respectively. It obtains the best
    speed/accuracy tradeoff, but the other two are still useful for research.
  * __C4__: Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
  * __DC5__ (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads
    for mask and box prediction, respectively.
    This is used by the Deformable ConvNet paper.
* Most models are trained with the 3x schedule (~37 COCO epochs).
  Although 1x models are heavily under-trained, we provide some ResNet-50 models with the 1x (~12 COCO epochs)
  training schedule for comparison when doing quick research iteration.

#### ImageNet Pretrained Models

We provide backbone models pretrained on ImageNet-1k dataset.
These models have __different__ format from those provided in Detectron: we do not fuse BatchNorm into an affine layer.
* [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
* [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
* [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.

Pretrained models in Detectron's format can still be used. For example:
* [X-152-32x8d-IN5k.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl):
  ResNeXt-152-32x8d model trained on ImageNet-5k with Caffe2 at FB (see ResNeXt paper for details on ImageNet-5k).
* [R-50-GN.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl):
  ResNet-50 with Group Normalization.
* [R-101-GN.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl):
  ResNet-101 with Group Normalization.

Torchvision's ResNet models can be used after converted by [this script](tools/convert-torchvision-to-d2.py).

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

### COCO Object Detection Baselines

#### Faster R-CNN:
<!--
(fb only) To update the table in vim:
1. Remove the old table: d}
2. Copy the below command to the place of the table
3. :.!bash

./gen_html_table.py --config 'COCO-Detection/faster*50*'{1x,3x}'*' 'COCO-Detection/faster*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml">R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.551</td>
<td align="center">0.102</td>
<td align="center">4.8</td>
<td align="center">35.7</td>
<td align="center">137257644</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml">R50-DC5</a></td>
<td align="center">1x</td>
<td align="center">0.380</td>
<td align="center">0.068</td>
<td align="center">5.0</td>
<td align="center">37.3</td>
<td align="center">137847829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.210</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">37.9</td>
<td align="center">137257794</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml">R50-C4</a></td>
<td align="center">3x</td>
<td align="center">0.543</td>
<td align="center">0.104</td>
<td align="center">4.8</td>
<td align="center">38.4</td>
<td align="center">137849393</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml">R50-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.378</td>
<td align="center">0.070</td>
<td align="center">5.0</td>
<td align="center">39.0</td>
<td align="center">137849425</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.209</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">40.2</td>
<td align="center">137849458</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml">R101-C4</a></td>
<td align="center">3x</td>
<td align="center">0.619</td>
<td align="center">0.139</td>
<td align="center">5.9</td>
<td align="center">41.1</td>
<td align="center">138204752</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml">R101-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.452</td>
<td align="center">0.086</td>
<td align="center">6.1</td>
<td align="center">40.6</td>
<td align="center">138204841</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.286</td>
<td align="center">0.051</td>
<td align="center">4.1</td>
<td align="center">42.0</td>
<td align="center">137851257</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.638</td>
<td align="center">0.098</td>
<td align="center">6.7</td>
<td align="center">43.0</td>
<td align="center">139173657</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/metrics.json">metrics</a></td>
</tr>
</tbody></table>

#### RetinaNet:
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml">R50</a></td>
<td align="center">1x</td>
<td align="center">0.205</td>
<td align="center">0.056</td>
<td align="center">4.1</td>
<td align="center">37.4</td>
<td align="center">190397773</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml">R50</a></td>
<td align="center">3x</td>
<td align="center">0.205</td>
<td align="center">0.056</td>
<td align="center">4.1</td>
<td align="center">38.7</td>
<td align="center">190397829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml">R101</a></td>
<td align="center">3x</td>
<td align="center">0.291</td>
<td align="center">0.069</td>
<td align="center">5.2</td>
<td align="center">40.4</td>
<td align="center">190397697</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/metrics.json">metrics</a></td>
</tr>
</tbody></table>


#### RPN & Fast R-CNN:
<!--
./gen_html_table.py --config 'COCO-Detection/rpn*' 'COCO-Detection/fast_rcnn*' --name "RPN R50-C4" "RPN R50-FPN" "Fast R-CNN R50-FPN" --fields lr_sched train_speed inference_speed mem box_AP prop_AR
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">prop.<br/>AR</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: rpn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_C4_1x.yaml">RPN R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.130</td>
<td align="center">0.034</td>
<td align="center">1.5</td>
<td align="center"></td>
<td align="center">51.6</td>
<td align="center">137258005</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/metrics.json">metrics</a></td>
</tr>
<!-- ROW: rpn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_FPN_1x.yaml">RPN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.186</td>
<td align="center">0.032</td>
<td align="center">2.7</td>
<td align="center"></td>
<td align="center">58.0</td>
<td align="center">137258492</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/metrics.json">metrics</a></td>
</tr>
<!-- ROW: fast_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml">Fast R-CNN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.140</td>
<td align="center">0.029</td>
<td align="center">2.6</td>
<td align="center">37.8</td>
<td align="center"></td>
<td align="center">137635226</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/metrics.json">metrics</a></td>
</tr>
</tbody></table>

### COCO Instance Segmentation Baselines with Mask R-CNN
<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask*50*'{1x,3x}'*' 'COCO-InstanceSegmentation/mask*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP
-->



<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml">R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.584</td>
<td align="center">0.110</td>
<td align="center">5.2</td>
<td align="center">36.8</td>
<td align="center">32.2</td>
<td align="center">137259246</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml">R50-DC5</a></td>
<td align="center">1x</td>
<td align="center">0.471</td>
<td align="center">0.076</td>
<td align="center">6.5</td>
<td align="center">38.3</td>
<td align="center">34.2</td>
<td align="center">137260150</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">38.6</td>
<td align="center">35.2</td>
<td align="center">137260431</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml">R50-C4</a></td>
<td align="center">3x</td>
<td align="center">0.575</td>
<td align="center">0.111</td>
<td align="center">5.2</td>
<td align="center">39.8</td>
<td align="center">34.4</td>
<td align="center">137849525</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml">R50-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.470</td>
<td align="center">0.076</td>
<td align="center">6.5</td>
<td align="center">40.0</td>
<td align="center">35.9</td>
<td align="center">137849551</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">41.0</td>
<td align="center">37.2</td>
<td align="center">137849600</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml">R101-C4</a></td>
<td align="center">3x</td>
<td align="center">0.652</td>
<td align="center">0.145</td>
<td align="center">6.3</td>
<td align="center">42.6</td>
<td align="center">36.7</td>
<td align="center">138363239</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml">R101-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.545</td>
<td align="center">0.092</td>
<td align="center">7.6</td>
<td align="center">41.9</td>
<td align="center">37.3</td>
<td align="center">138363294</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.340</td>
<td align="center">0.056</td>
<td align="center">4.6</td>
<td align="center">42.9</td>
<td align="center">38.6</td>
<td align="center">138205316</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.690</td>
<td align="center">0.103</td>
<td align="center">7.2</td>
<td align="center">44.3</td>
<td align="center">39.5</td>
<td align="center">139653917</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a></td>
</tr>
</tbody></table>

### COCO Person Keypoint Detection Baselines with Keypoint R-CNN
<!--
./gen_html_table.py --config 'COCO-Keypoints/*50*' 'COCO-Keypoints/*101*'  --name R50-FPN R50-FPN R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP keypoint_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">kp.<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: keypoint_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.315</td>
<td align="center">0.072</td>
<td align="center">5.0</td>
<td align="center">53.6</td>
<td align="center">64.0</td>
<td align="center">137261548</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/metrics.json">metrics</a></td>
</tr>
<!-- ROW: keypoint_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.316</td>
<td align="center">0.066</td>
<td align="center">5.0</td>
<td align="center">55.4</td>
<td align="center">65.5</td>
<td align="center">137849621</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/metrics.json">metrics</a></td>
</tr>
<!-- ROW: keypoint_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.390</td>
<td align="center">0.076</td>
<td align="center">6.1</td>
<td align="center">56.4</td>
<td align="center">66.1</td>
<td align="center">138363331</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/metrics.json">metrics</a></td>
</tr>
<!-- ROW: keypoint_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.738</td>
<td align="center">0.121</td>
<td align="center">8.7</td>
<td align="center">57.3</td>
<td align="center">66.0</td>
<td align="center">139686956</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/metrics.json">metrics</a></td>
</tr>
</tbody></table>

### COCO Panoptic Segmentation Baselines with Panoptic FPN
<!--
./gen_html_table.py --config 'COCO-PanopticSegmentation/*50*' 'COCO-PanopticSegmentation/*101*'  --name R50-FPN R50-FPN R101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP PQ
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">PQ</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: panoptic_fpn_R_50_1x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.304</td>
<td align="center">0.053</td>
<td align="center">4.8</td>
<td align="center">37.6</td>
<td align="center">34.7</td>
<td align="center">39.4</td>
<td align="center">139514544</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R_50_3x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.302</td>
<td align="center">0.053</td>
<td align="center">4.8</td>
<td align="center">40.0</td>
<td align="center">36.5</td>
<td align="center">41.5</td>
<td align="center">139514569</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/metrics.json">metrics</a></td>
</tr>
<!-- ROW: panoptic_fpn_R_101_3x -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.392</td>
<td align="center">0.066</td>
<td align="center">6.0</td>
<td align="center">42.4</td>
<td align="center">38.5</td>
<td align="center">43.0</td>
<td align="center">139514519</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/metrics.json">metrics</a></td>
</tr>
</tbody></table>


### LVIS Instance Segmentation Baselines with Mask R-CNN

Mask R-CNN baselines on the [LVIS dataset](https://lvisdataset.org), v0.5.
These baselines are described in Table 3(c) of the [LVIS paper](https://arxiv.org/abs/1908.03195).

NOTE: the 1x schedule here has the same amount of __iterations__ as the COCO 1x baselines.
They are roughly 24 epochs of LVISv0.5 data.
The final results of these configs have large variance across different runs.

<!--
./gen_html_table.py --config 'LVISv0.5-InstanceSegmentation/mask*50*' 'LVISv0.5-InstanceSegmentation/mask*101*' --name R50-FPN R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.292</td>
<td align="center">0.107</td>
<td align="center">7.1</td>
<td align="center">23.6</td>
<td align="center">24.4</td>
<td align="center">144219072</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_FPN_1x -->
 <tr><td align="left"><a href="configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml">R101-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.371</td>
<td align="center">0.114</td>
<td align="center">7.8</td>
<td align="center">25.6</td>
<td align="center">25.9</td>
<td align="center">144219035</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/model_final_824ab5.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_X_101_32x8d_FPN_1x -->
 <tr><td align="left"><a href="configs/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml">X101-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.712</td>
<td align="center">0.151</td>
<td align="center">10.2</td>
<td align="center">26.7</td>
<td align="center">27.1</td>
<td align="center">144219108</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/metrics.json">metrics</a></td>
</tr>
</tbody></table>



### Cityscapes & Pascal VOC Baselines

Simple baselines for
* Mask R-CNN on Cityscapes instance segmentation (initialized from COCO pre-training, then trained on Cityscapes fine annotations only)
* Faster R-CNN on PASCAL VOC object detection (trained on VOC 2007 train+val + VOC 2012 train+val, tested on VOC 2007 using 11-point interpolated AP)

<!--
./gen_html_table.py --config 'Cityscapes/*' 'PascalVOC-Detection/*' --name "R50-FPN, Cityscapes" "R50-C4, VOC" --fields train_speed inference_speed mem box_AP box_AP50 mask_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">box<br/>AP50</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_FPN -->
 <tr><td align="left"><a href="configs/Cityscapes/mask_rcnn_R_50_FPN.yaml">R50-FPN, Cityscapes</a></td>
<td align="center">0.240</td>
<td align="center">0.078</td>
<td align="center">4.4</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">36.5</td>
<td align="center">142423278</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_C4 -->
 <tr><td align="left"><a href="configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml">R50-C4, VOC</a></td>
<td align="center">0.537</td>
<td align="center">0.081</td>
<td align="center">4.8</td>
<td align="center">51.9</td>
<td align="center">80.3</td>
<td align="center"></td>
<td align="center">142202221</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/model_final_b1acc2.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/metrics.json">metrics</a></td>
</tr>
</tbody></table>



### Other Settings

Ablations for Deformable Conv and Cascade R-CNN:

<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml' 'Misc/*R_50_FPN_1x_dconv*' 'Misc/cascade*1x.yaml' 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' 'Misc/*R_50_FPN_3x_dconv*' 'Misc/cascade*3x.yaml' --name "Baseline R50-FPN" "Deformable Conv" "Cascade R-CNN" "Baseline R50-FPN" "Deformable Conv" "Cascade R-CNN"  --fields lr_sched train_speed inference_speed mem box_AP mask_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">Baseline R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">38.6</td>
<td align="center">35.2</td>
<td align="center">137260431</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_1x_dconv_c3-c5 -->
 <tr><td align="left"><a href="configs/Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml">Deformable Conv</a></td>
<td align="center">1x</td>
<td align="center">0.342</td>
<td align="center">0.048</td>
<td align="center">3.5</td>
<td align="center">41.5</td>
<td align="center">37.5</td>
<td align="center">138602867</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5/138602867/model_final_65c703.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5/138602867/metrics.json">metrics</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml">Cascade R-CNN</a></td>
<td align="center">1x</td>
<td align="center">0.317</td>
<td align="center">0.052</td>
<td align="center">4.0</td>
<td align="center">42.1</td>
<td align="center">36.4</td>
<td align="center">138602847</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_1x/138602847/model_final_e9d89b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_1x/138602847/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml">Baseline R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">41.0</td>
<td align="center">37.2</td>
<td align="center">137849600</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x_dconv_c3-c5 -->
 <tr><td align="left"><a href="configs/Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml">Deformable Conv</a></td>
<td align="center">3x</td>
<td align="center">0.349</td>
<td align="center">0.047</td>
<td align="center">3.5</td>
<td align="center">42.7</td>
<td align="center">38.5</td>
<td align="center">144998336</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5/144998336/model_final_821d0b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5/144998336/metrics.json">metrics</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml">Cascade R-CNN</a></td>
<td align="center">3x</td>
<td align="center">0.328</td>
<td align="center">0.053</td>
<td align="center">4.0</td>
<td align="center">44.3</td>
<td align="center">38.5</td>
<td align="center">144998488</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/metrics.json">metrics</a></td>
</tr>
</tbody></table>


Ablations for normalization methods, and a few models trained from scratch following [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883).
(Note: The baseline uses `2fc` head while the others use [`4conv1fc` head](https://arxiv.org/abs/1803.08494))
<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' 'Misc/mask*50_FPN_3x_gn.yaml' 'Misc/mask*50_FPN_3x_syncbn.yaml' 'Misc/scratch*' --name "Baseline R50-FPN" "GN" "SyncBN" "GN (from scratch)" "GN (from scratch)" "SyncBN (from scratch)" --fields lr_sched train_speed inference_speed mem box_AP mask_AP
   -->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml">Baseline R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">41.0</td>
<td align="center">37.2</td>
<td align="center">137849600</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x_gn -->
 <tr><td align="left"><a href="configs/Misc/mask_rcnn_R_50_FPN_3x_gn.yaml">GN</a></td>
<td align="center">3x</td>
<td align="center">0.309</td>
<td align="center">0.060</td>
<td align="center">5.6</td>
<td align="center">42.6</td>
<td align="center">38.6</td>
<td align="center">138602888</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_gn/138602888/model_final_dc5d9e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_gn/138602888/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x_syncbn -->
 <tr><td align="left"><a href="configs/Misc/mask_rcnn_R_50_FPN_3x_syncbn.yaml">SyncBN</a></td>
<td align="center">3x</td>
<td align="center">0.345</td>
<td align="center">0.053</td>
<td align="center">5.5</td>
<td align="center">41.9</td>
<td align="center">37.8</td>
<td align="center">169527823</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_syncbn/169527823/model_final_3b3c51.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/mask_rcnn_R_50_FPN_3x_syncbn/169527823/metrics.json">metrics</a></td>
</tr>
<!-- ROW: scratch_mask_rcnn_R_50_FPN_3x_gn -->
 <tr><td align="left"><a href="configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml">GN (from scratch)</a></td>
<td align="center">3x</td>
<td align="center">0.338</td>
<td align="center">0.061</td>
<td align="center">7.2</td>
<td align="center">39.9</td>
<td align="center">36.6</td>
<td align="center">138602908</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn/138602908/model_final_01ca85.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn/138602908/metrics.json">metrics</a></td>
</tr>
<!-- ROW: scratch_mask_rcnn_R_50_FPN_9x_gn -->
 <tr><td align="left"><a href="configs/Misc/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml">GN (from scratch)</a></td>
<td align="center">9x</td>
<td align="center">N/A</td>
<td align="center">0.061</td>
<td align="center">7.2</td>
<td align="center">43.7</td>
<td align="center">39.6</td>
<td align="center">183808979</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_9x_gn/183808979/model_final_da7b4c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_9x_gn/183808979/metrics.json">metrics</a></td>
</tr>
<!-- ROW: scratch_mask_rcnn_R_50_FPN_9x_syncbn -->
 <tr><td align="left"><a href="configs/Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml">SyncBN (from scratch)</a></td>
<td align="center">9x</td>
<td align="center">N/A</td>
<td align="center">0.055</td>
<td align="center">7.2</td>
<td align="center">43.6</td>
<td align="center">39.3</td>
<td align="center">184226666</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn/184226666/model_final_5ce33e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn/184226666/metrics.json">metrics</a></td>
</tr>
</tbody></table>


A few very large models trained for a long time, for demo purposes. They are trained using multiple machines:

<!--
./gen_html_table.py --config 'Misc/panoptic_*dconv*' 'Misc/cascade_*152*' --name "Panoptic FPN R101" "Mask R-CNN X152" --fields inference_speed mem box_AP mask_AP PQ
# manually add TTA results
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">PQ</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: panoptic_fpn_R_101_dconv_cascade_gn_3x -->
 <tr><td align="left"><a href="configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml">Panoptic FPN R101</a></td>
<td align="center">0.098</td>
<td align="center">11.4</td>
<td align="center">47.4</td>
<td align="center">41.3</td>
<td align="center">46.1</td>
<td align="center">139797668</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/metrics.json">metrics</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv -->
 <tr><td align="left"><a href="configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml">Mask R-CNN X152</a></td>
<td align="center">0.234</td>
<td align="center">15.1</td>
<td align="center">50.2</td>
<td align="center">44.0</td>
<td align="center"></td>
<td align="center">18131413</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/metrics.json">metrics</a></td>
</tr>
<!-- ROW: TTA cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv -->
 <tr><td align="left">above + test-time aug.</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">51.9</td>
<td align="center">45.9</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
</tr>
</tbody></table>
