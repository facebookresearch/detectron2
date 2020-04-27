
Detectron2 model zoo's experimental settings and a few implementation details are different from Detectron.

The differences in implementation details are shared in
[Compatibility with Other Libraries](../../docs/notes/compatibility.md).

The differences in model zoo's experimental settings include:
* Use scale augmentation during training. This improves AP with lower training cost.
* Use L1 loss instead of smooth L1 loss for simplicity. This sometimes improves box AP but may
  affect other AP.
* Use `POOLER_SAMPLING_RATIO=0` instead of 2. This does not significantly affect AP.
* Use `ROIAlignV2`. This does not significantly affect AP.

In this directory, we provide a few configs that __do not__ have the above changes.
They mimic Detectron's behavior as close as possible,
and provide a fair comparison of accuracy and speed against Detectron.

<!--
./gen_html_table.py --config 'Detectron1-Comparisons/*.yaml' --name "Faster R-CNN" "Keypoint R-CNN" "Mask R-CNN" --fields lr_sched train_speed inference_speed mem box_AP mask_AP keypoint_AP --base-dir ../../../configs/Detectron1-Comparisons
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
<th valign="bottom">kp.<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_FPN_noaug_1x -->
 <tr><td align="left"><a href="faster_rcnn_R_50_FPN_noaug_1x.yaml">Faster R-CNN</a></td>
<td align="center">1x</td>
<td align="center">0.219</td>
<td align="center">0.038</td>
<td align="center">3.1</td>
<td align="center">36.9</td>
<td align="center"></td>
<td align="center"></td>
<td align="center">137781054</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x/137781054/model_final_7ab50c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x/137781054/metrics.json">metrics</a></td>
</tr>
<!-- ROW: keypoint_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="keypoint_rcnn_R_50_FPN_1x.yaml">Keypoint R-CNN</a></td>
<td align="center">1x</td>
<td align="center">0.313</td>
<td align="center">0.071</td>
<td align="center">5.0</td>
<td align="center">53.1</td>
<td align="center"></td>
<td align="center">64.2</td>
<td align="center">137781195</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x/137781195/model_final_cce136.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x/137781195/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_noaug_1x -->
 <tr><td align="left"><a href="mask_rcnn_R_50_FPN_noaug_1x.yaml">Mask R-CNN</a></td>
<td align="center">1x</td>
<td align="center">0.273</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">37.8</td>
<td align="center">34.9</td>
<td align="center"></td>
<td align="center">137781281</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x/137781281/model_final_62ca52.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x/137781281/metrics.json">metrics</a></td>
</tr>
</tbody></table>

## Comparisons:

* Faster R-CNN: Detectron's AP is 36.7, similar to ours.
* Keypoint R-CNN: Detectron's AP is box 53.6, keypoint 64.2. Fixing a Detectron's
  [bug](https://github.com/facebookresearch/Detectron/issues/459) lead to a drop in box AP, and can be
	compensated back by some parameter tuning.
* Mask R-CNN: Detectron's AP is box 37.7, mask 33.9. We're 1 AP better in mask AP, due to more correct implementation.

For speed comparison, see [benchmarks](https://detectron2.readthedocs.io/notes/benchmarks.html).
