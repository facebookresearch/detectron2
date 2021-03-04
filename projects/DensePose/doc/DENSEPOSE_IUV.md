# Chart-based Dense Pose Estimation for Humans and Animals

## <a name="Overview"></a> Overview

The goal of chart-based DensePose methods is to establish dense correspondences
between image pixels and 3D object mesh by splitting the latter into charts and estimating
for each pixel the corresponding chart index `I` and local chart coordinates `(U, V)`.

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_teaser.gif" width="700px" />
</div>

The charts used for human DensePose estimation are shown in Figure 1.
The human body is split into 24 parts, each part is parametrized by `U` and `V`
coordinates, each taking values in `[0, 1]`.

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/coords.png" width="500px" />
</div>
<p class="image-caption">Figure 1. Partitioning and parametrization of human body surface.</p>

The pipeline uses [Faster R-CNN](https://arxiv.org/abs/1506.01497)
with [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) meta architecture
outlined in Figure 2. For each detected object, the model predicts
its coarse segmentation `S` (2 or 15 channels: foreground / background or
background + 14 predefined body parts), fine segmentation `I` (25 channels:
background + 24 predefined body parts) and local chart coordinates `U` and `V`.

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_pipeline_iuv.png" width="500px" />
</div>
<p class="image-caption">Figure 2. DensePose chart-based architecture based on Faster R-CNN with Feature Pyramid Network (FPN).</p>

## <a name="ModelZoo"></a> Model Zoo and Baselines

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

### Improved Baselines, Original Fully Convolutional Head

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

### Baselines with Confidence Estimation

These models perform additional estimation of confidence in regressed UV coodrinates, along the lines of [Neverova et al., 2019](https://papers.nips.cc/paper/8378-correlated-uncertainty-for-learning-dense-correspondences-from-noisy-labels).

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
<!-- ROW: densepose_rcnn_R_50_FPN_WC1_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_WC1_s1x.yaml">R_50_FPN_WC1_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.353</td>
<td align="center">0.064</td>
<td align="center">4.6</td>
<td align="center">60.5</td>
<td align="center">64.2</td>
<td align="center">65.6</td>
<td align="center">173862049</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1_s1x/173862049/model_final_289019.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1_s1x/173862049/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_WC2_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_WC2_s1x.yaml">R_50_FPN_WC2_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.364</td>
<td align="center">0.066</td>
<td align="center">4.8</td>
<td align="center">60.7</td>
<td align="center">64.2</td>
<td align="center">65.7</td>
<td align="center">173861455</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC2_s1x/173861455/model_final_3abe14.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC2_s1x/173861455/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_DL_WC1_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_DL_WC1_s1x.yaml">R_50_FPN_DL_WC1_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.397</td>
<td align="center">0.068</td>
<td align="center">6.7</td>
<td align="center">61.1</td>
<td align="center">65.8</td>
<td align="center">67.1</td>
<td align="center">173067973</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC1_s1x/173067973/model_final_b1e525.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC1_s1x/173067973/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_DL_WC2_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_DL_WC2_s1x.yaml">R_50_FPN_DL_WC2_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.410</td>
<td align="center">0.070</td>
<td align="center">6.8</td>
<td align="center">60.8</td>
<td align="center">65.6</td>
<td align="center">66.7</td>
<td align="center">173859335</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC2_s1x/173859335/model_final_60fed4.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC2_s1x/173859335/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_WC1_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_WC1_s1x.yaml">R_101_FPN_WC1_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.435</td>
<td align="center">0.076</td>
<td align="center">5.7</td>
<td align="center">62.5</td>
<td align="center">64.9</td>
<td align="center">66.5</td>
<td align="center">171402969</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC1_s1x/171402969/model_final_9e47f0.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC1_s1x/171402969/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_WC2_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_WC2_s1x.yaml">R_101_FPN_WC2_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.450</td>
<td align="center">0.078</td>
<td align="center">5.7</td>
<td align="center">62.3</td>
<td align="center">64.8</td>
<td align="center">66.6</td>
<td align="center">173860702</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC2_s1x/173860702/model_final_5ea023.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC2_s1x/173860702/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_DL_WC1_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml">R_101_FPN_DL_WC1_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.479</td>
<td align="center">0.081</td>
<td align="center">7.9</td>
<td align="center">62.0</td>
<td align="center">66.2</td>
<td align="center">67.4</td>
<td align="center">173858525</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1_s1x/173858525/model_final_f359f3.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1_s1x/173858525/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_DL_WC2_s1x -->
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml">R_101_FPN_DL_WC2_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.491</td>
<td align="center">0.082</td>
<td align="center">7.6</td>
<td align="center">61.7</td>
<td align="center">65.9</td>
<td align="center">67.3</td>
<td align="center">173294801</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/metrics.json">metrics</a></td>
</tr>
</tbody></table>

### License

All models available for download are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/)

## <a name="References"></a> References

If you use chart-based DensePose methods, please take the references from the following
BibTeX entries:

DensePose bootstrapping pipeline:
```
@InProceedings{Sanakoyeu2020TransferringDensePose,
    title = {Transferring Dense Pose to Proximal Animal Classes},
    author = {Artsiom Sanakoyeu and Vasil Khalidov and Maureen S. McCarthy and Andrea Vedaldi and Natalia Neverova},
    journal = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020},
}
```

DensePose with confidence estimation:
```
@InProceedings{Neverova2019DensePoseConfidences,
    title = {Correlated Uncertainty for Learning Dense Correspondences from Noisy Labels},
    author = {Neverova, Natalia and Novotny, David and Vedaldi, Andrea},
    journal = {Advances in Neural Information Processing Systems},
    year = {2019},
}
```

Original DensePose:
```
@InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
