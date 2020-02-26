# Model Zoo and Baselines

# Introduction

We provide baselines trained with Detectron2 DensePose. The corresponding
configuration files can be found in the [configs](../configs) directory.
All models were trained on COCO `train2014` + `valminusminival2014` and
evaluated on COCO `minival2014`. For the details on common settings in which
baselines were trained, please check [Detectron 2 Model Zoo](../../../MODEL_ZOO.md).

## License

All models available for download through this document are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/)

## COCO DensePose Baselines with DensePose-RCNN

### Legacy Models

Baselines trained using schedules from [Güler et al, 2018](https://arxiv.org/pdf/1802.00434.pdf)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_s1x_legacy -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml">R_50_FPN_s1x_legacy</a></td>
 <td align="center">s1x</td>
 <td align="center">0.307</td>
 <td align="center">0.051</td>
 <td align="center">3.2</td>
 <td align="center">58.1</td>
 <td align="center">52.1</td>
 <td align="center">54.9</td>
 <td align="center">164832157</td>
 <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/metrics.json">metrics</a></td>
 </tr>
 <!-- ROW: densepose_rcnn_R_101_FPN_s1x_legacy -->
  <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml">R_101_FPN_s1x_legacy</a></td>
  <td align="center">s1x</td>
  <td align="center">0.390</td>
  <td align="center">0.063</td>
  <td align="center">4.3</td>
  <td align="center">59.5</td>
  <td align="center">53.2</td>
  <td align="center">56.1</td>
  <td align="center">164832182</td>
  <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/metrics.json">metrics</a></td>
  </tr>
</tbody></table>

### Improved Baselines, Original Fully Convolutional Haad

These models use an improved training schedule and Panoptic FPN head from [Kirillov et al, 2019](https://arxiv.org/abs/1901.02446).

<table><tbody>
  <!-- START TABLE -->
  <!-- TABLE HEADER -->
  <th valign="bottom">Name</th>
  <th valign="bottom">lr<br/>sched</th>
  <th valign="bottom">train<br/>time<br/>(s/iter)</th>
  <th valign="bottom">inference<br/>time<br/>(s/im)</th>
  <th valign="bottom">train<br/>mem<br/>(GB)</th>
  <th valign="bottom">box<br/>AP</th>
  <th valign="bottom">dp. AP<br/>GPS</th>
  <th valign="bottom">dp. AP<br/>GPSm</th>
  <th valign="bottom">model id</th>
  <th valign="bottom">download</th>
  <!-- TABLE BODY -->
  <!-- ROW: densepose_rcnn_R_50_FPN_s1x -->
   <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_s1x.yaml">R_50_FPN_s1x</a></td>
   <td align="center">s1x</td>
   <td align="center">0.359</td>
   <td align="center">0.066</td>
   <td align="center">4.5</td>
   <td align="center">61.2</td>
   <td align="center">63.7</td>
   <td align="center">65.3</td>
   <td align="center">165712039</td>
   <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/metrics.json">metrics</a></td>
   </tr>
   <!-- ROW: densepose_rcnn_R_101_FPN_s1x -->
    <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_s1x.yaml">R_101_FPN_s1x</a></td>
    <td align="center">s1x</td>
    <td align="center">0.428</td>
    <td align="center">0.079</td>
    <td align="center">5.8</td>
    <td align="center">62.3</td>
    <td align="center">64.5</td>
    <td align="center">66.4</td>
    <td align="center">165712084</td>
    <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/metrics.json">metrics</a></td>
    </tr>
    </tbody></table>

### Improved Baselines, DeepLabV3 Head

These models use an improved training schedule, Panoptic FPN head from [Kirillov et al, 2019](https://arxiv.org/abs/1901.02446) and DeepLabV3 head from [Chen et al, 2017](https://arxiv.org/abs/1706.05587).

<table><tbody>
    <!-- START TABLE -->
    <!-- TABLE HEADER -->
    <th valign="bottom">Name</th>
    <th valign="bottom">lr<br/>sched</th>
    <th valign="bottom">train<br/>time<br/>(s/iter)</th>
    <th valign="bottom">inference<br/>time<br/>(s/im)</th>
    <th valign="bottom">train<br/>mem<br/>(GB)</th>
    <th valign="bottom">box<br/>AP</th>
    <th valign="bottom">dp. AP<br/>GPS</th>
    <th valign="bottom">dp. AP<br/>GPSm</th>
    <th valign="bottom">model id</th>
    <th valign="bottom">download</th>
    <!-- TABLE BODY -->
    <!-- ROW: densepose_rcnn_R_50_FPN_DL_s1x -->
     <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml">R_50_FPN_DL_s1x</a></td>
     <td align="center">s1x</td>
     <td align="center">0.392</td>
     <td align="center">0.070</td>
     <td align="center">6.7</td>
     <td align="center">61.1</td>
     <td align="center">65.6</td>
     <td align="center">66.8</td>
     <td align="center">165712097</td>
     <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/metrics.json">metrics</a></td>
     </tr>
     <!-- ROW: densepose_rcnn_R_101_FPN_DL_s1x -->
      <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml">R_101_FPN_DL_s1x</a></td>
      <td align="center">s1x</td>
      <td align="center">0.478</td>
      <td align="center">0.083</td>
      <td align="center">7.0</td>
      <td align="center">62.3</td>
      <td align="center">66.3</td>
      <td align="center">67.7</td>
      <td align="center">165712116</td>
      <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/metrics.json">metrics</a></td>
      </tr>
</tbody></table>

## Old Baselines

It is still possible to use some baselines from [DensePose 1](https://github.com/facebookresearch/DensePose).
Below are evaluation metrics for the baselines recomputed in the current framework:

| Model | bbox AP | AP  |  AP50 | AP75  | APm  |APl |
|-----|-----|-----|---    |---    |---   |--- |
| [`ResNet50_FPN_s1x-e2e`](https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet50_FPN_s1x-e2e.pkl) | 54.673 | 48.894 | 84.963 | 50.717 | 43.132 | 50.433 |
| [`ResNet101_FPN_s1x-e2e`](https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl) | 56.032 | 51.088 | 86.250 | 55.057 | 46.542 | 52.563 |

Note: these scores are close, but not strictly equal to the ones reported in the [DensePose 1 Model Zoo](https://github.com/facebookresearch/DensePose/blob/master/MODEL_ZOO.md),
which is due to small incompatibilities between the frameworks.
