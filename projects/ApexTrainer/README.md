# Apex Trainer Example


## Install Apex

- apex: `git clone https://github.com/NVIDIA/apex; cd apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## RetinaNet:
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
<td align="center">0.200</td>
<td align="center">0.062</td>
<td align="center">3.9</td>
<td align="center">36.5</td>
<td align="center">137593951</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/137593951/model_final_b796dc.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/137593951/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml">R50</a></td>
<td align="center">3x</td>
<td align="center">0.201</td>
<td align="center">0.063</td>
<td align="center">3.9</td>
<td align="center">37.9</td>
<td align="center">137849486</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/137849486/model_final_4cafe0.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/137849486/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_Apex_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_Apex_1x.yaml">R50_ApexTrainer</a></td>
<td align="center">1x</td>
<td align="center">0.423</td>
<td align="center">0.063</td>
<td align="center">5.8</td>
<td align="center">0.367</td>
<td align="center">-</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/138363263/metrics.json">metrics</a></td>
</tr>
</tbody></table>