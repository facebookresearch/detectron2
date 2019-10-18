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

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">dp.<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY --> 
<!-- ROW: densepose_rcnn_R_50_FPN_s1x --> 
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_s1x.yaml">R_50_FPN_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.281</td>
<td align="center">0.064</td>
<td align="center">3.2</td>
<td align="center">57.8</td>
<td align="center">49.8</td>
<td align="center">143908701</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/143908701/model_final_dd99d2.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/143908701/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_s1x --> 
 <tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_s1x.yaml">R_101_FPN_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.365</td>
<td align="center">0.076</td>
<td align="center">4.3</td>
<td align="center">59.5</td>
<td align="center">51.1</td>
<td align="center">143908726</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/143908726/model_final_ad63b5.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/143908726/metrics.json">metrics</a></td>
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
