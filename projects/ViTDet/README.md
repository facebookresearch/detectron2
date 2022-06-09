# ViTDet: Exploring Plain Vision Transformer Backbones for Object Detection

Yanghao Li, Hanzi Mao, Ross Girshick†, Kaiming He†

[[`arXiv`](https://arxiv.org/abs/2203.16527)] [[`BibTeX`](#CitingViTDet)]

In this repository, we provide configs and models in Detectron2 for ViTDet as well as MViTv2 and Swin backbones with our implementation and settings as described in [ViTDet](https://arxiv.org/abs/2203.16527) paper.


## Pretrained Models

### COCO

#### Mask R-CNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">train<br/>time<br/>(s/im)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_vitdet_b_100ep -->
 <tr><td align="left"><a href="configs/COCO/mask_rcnn_vitdet_b_100ep.py">ViTDet, ViT-B</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.314</td>
<td align="center">0.079</td>
<td align="center">10.9</td>
<td align="center">51.6</td>
<td align="center">45.9</td>
<td align="center">325346929</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl">model</a></td>
</tr>
<!-- ROW: mask_rcnn_vitdet_l_100ep -->
 <tr><td align="left"><a href="configs/COCO/mask_rcnn_vitdet_l_100ep.py">ViTDet, ViT-L</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.603</td>
<td align="center">0.125</td>
<td align="center">20.9</td>
<td align="center">55.5</td>
<td align="center">49.2</td>
<td align="center">325599698</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_l/f325599698/model_final_6146ed.pkl">model</a></td>
</tr>
<!-- ROW: mask_rcnn_vitdet_b_75ep -->
 <tr><td align="left"><a href="configs/COCO/mask_rcnn_vitdet_h_75ep.py">ViTDet, ViT-H</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">1.098</td>
<td align="center">0.178</td>
<td align="center">31.5</td>
<td align="center">56.7</td>
<td align="center">50.2</td>
<td align="center">329145471</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_h/f329145471/model_final_7224f1.pkl">model</a></td>
</tr>
</tbody></table>

#### Cascade Mask R-CNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">train<br/>time<br/>(s/im)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: cascade_mask_rcnn_swin_b_in21k_50ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep.py">Swin-B</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.389</td>
<td align="center">0.077</td>
<td align="center">8.7</td>
<td align="center">53.9</td>
<td align="center">46.2</td>
<td align="center">342979038</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_swin_b_in21k/f342979038/model_final_246a82.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_swin_l_in21k_50ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_swin_l_in21k_50ep.py">Swin-L</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.508</td>
<td align="center">0.097</td>
<td align="center">12.6</td>
<td align="center">55.0</td>
<td align="center">47.2</td>
<td align="center">342979186</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_swin_l_in21k/f342979186/model_final_7c897e.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_b_in21k_100ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py">MViTv2-B</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.475</td>
<td align="center">0.090</td>
<td align="center">8.9</td>
<td align="center">55.6</td>
<td align="center">48.1</td>
<td align="center">325820315</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_mvitv2_b_in21k/f325820315/model_final_8c3da3.pkl">model</a></td>
</tr>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_l_in21k_50ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_mvitv2_l_in21k_50ep.py">MViTv2-L</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.844</td>
<td align="center">0.157</td>
<td align="center">19.7</td>
<td align="center">55.7</td>
<td align="center">48.3</td>
<td align="center">325607715</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_mvitv2_l_in21k/f325607715/model_final_2141b0.pkl">model</a></td>
</tr>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_h_in21k_36ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_mvitv2_h_in21k_36ep.py">MViTv2-H</a></td>
<td align="center">IN21K, sup</td>
<td align="center">1.655</td>
<td align="center">0.285</td>
<td align="center">18.4*</td>
<td align="center">55.9</td>
<td align="center">48.3</td>
<td align="center">326187358</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_mvitv2_h_in21k/f326187358/model_final_2234d7.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_b_100ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_vitdet_b_100ep.py">ViTDet, ViT-B</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.362</td>
<td align="center">0.089</td>
<td align="center">12.3</td>
<td align="center">54.0</td>
<td align="center">46.7</td>
<td align="center">325358525</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_l_100ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_vitdet_l_100ep.py">ViTDet, ViT-L</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.643</td>
<td align="center">0.142</td>
<td align="center">22.3</td>
<td align="center">57.6</td>
<td align="center">50.0</td>
<td align="center">328021305</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_l/f328021305/model_final_1a9f28.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_h_75ep -->
 <tr><td align="left"><a href="configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py">ViTDet, ViT-H</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">1.137</td>
<td align="center">0.196</td>
<td align="center">32.9</td>
<td align="center">58.7</td>
<td align="center">51.0</td>
<td align="center">328730692</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl">model</a></td>
</tr>
</tbody></table>


### LVIS

#### Mask R-CNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">train<br/>time<br/>(s/im)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_vitdet_b_100ep -->
 <tr><td align="left"><a href="configs/LVIS/mask_rcnn_vitdet_b_100ep.py">ViTDet, ViT-B</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.317</td>
<td align="center">0.085</td>
<td align="center">14.4</td>
<td align="center">40.2</td>
<td align="center">38.2</td>
<td align="center">329225748</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/mask_rcnn_vitdet_b/329225748/model_final_5251c5.pkl">model</a></td>
</tr>
<!-- ROW: mask_rcnn_vitdet_l_100ep -->
 <tr><td align="left"><a href="configs/LVIS/mask_rcnn_vitdet_l_100ep.py">ViTDet, ViT-L</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.576</td>
<td align="center">0.137</td>
<td align="center">24.7</td>
<td align="center">46.1</td>
<td align="center">43.6</td>
<td align="center">329211570</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/mask_rcnn_vitdet_l/329211570/model_final_021b3a.pkl">model</a></td>
</tr>
<!-- ROW: mask_rcnn_vitdet_b_75ep -->
 <tr><td align="left"><a href="configs/LVIS/mask_rcnn_vitdet_h_100ep.py">ViTDet, ViT-H</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">1.059</td>
<td align="center">0.186</td>
<td align="center">35.3</td>
<td align="center">49.1</td>
<td align="center">46.0</td>
<td align="center">332434656</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/mask_rcnn_vitdet_h/332434656/model_final_866730.pkl">model</a></td>
</tr>
</tbody></table>

#### Cascade Mask R-CNN

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">train<br/>time<br/>(s/im)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: cascade_mask_rcnn_swin_b_in21k_50ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_swin_b_in21k_50ep.py">Swin-B</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.368</td>
<td align="center">0.090</td>
<td align="center">11.5</td>
<td align="center">44.0</td>
<td align="center">39.6</td>
<td align="center">329222304</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_swin_b_in21k/329222304/model_final_a3a348.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_swin_l_in21k_50ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_swin_l_in21k_50ep.py">Swin-L</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.486</td>
<td align="center">0.105</td>
<td align="center">13.8</td>
<td align="center">46.0</td>
<td align="center">41.4</td>
<td align="center">329222724</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_swin_l_in21k/329222724/model_final_2b94db.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_b_in21k_100ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py">MViTv2-B</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.475</td>
<td align="center">0.100</td>
<td align="center">11.8</td>
<td align="center">46.3</td>
<td align="center">42.0</td>
<td align="center">329477206</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_mvitv2_b_in21k/329477206/model_final_a00567.pkl">model</a></td>
</tr>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_l_in21k_50ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_mvitv2_l_in21k_50ep.py">MViTv2-L</a></td>
<td align="center">IN21K, sup</td>
<td align="center">0.844</td>
<td align="center">0.172</td>
<td align="center">21.0</td>
<td align="center">49.4</td>
<td align="center">44.2</td>
<td align="center">329661552</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_mvitv2_l_in21k/329661552/model_final_7838a5.pkl">model</a></td>
</tr>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_h_in21k_36ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_mvitv2_h_in21k_50ep.py">MViTv2-H</a></td>
<td align="center">IN21K, sup</td>
<td align="center">1.661</td>
<td align="center">0.290</td>
<td align="center">21.3*</td>
<td align="center">49.5</td>
<td align="center">44.1</td>
<td align="center">330445165</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_mvitv2_h_in21k/330445165/model_final_ad4220.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_b_100ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_vitdet_b_100ep.py">ViTDet, ViT-B</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.356</td>
<td align="center">0.099</td>
<td align="center">15.2</td>
<td align="center">43.0</td>
<td align="center">38.9</td>
<td align="center">329226874</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_b/329226874/model_final_df306f.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_l_100ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_vitdet_l_100ep.py">ViTDet, ViT-L</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">0.629</td>
<td align="center">0.150</td>
<td align="center">24.9</td>
<td align="center">49.2</td>
<td align="center">44.5</td>
<td align="center">329042206</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_l/329042206/model_final_3e81c2.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_vitdet_h_75ep -->
 <tr><td align="left"><a href="configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py">ViTDet, ViT-H</a></td>
<td align="center">IN1K, MAE</td>
<td align="center">1.100</td>
<td align="center">0.204</td>
<td align="center">35.5</td>
<td align="center">51.5</td>
<td align="center">46.6</td>
<td align="center">332552778</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/ViTDet/LVIS/cascade_mask_rcnn_vitdet_h/332552778/model_final_11bbb7.pkl">model</a></td>
</tr>
</tbody></table>

Note: Unlike the system-level comparisons in the paper, these models use a lower resolution (1024 instead of 1280) and standard NMS (instead of soft NMS). As a result, they have slightly lower box and mask AP.

We observed higher variance on LVIS evalution results compared to COCO. For example, the standard deviations of box AP and mask AP were 0.30% (compared to 0.10% on COCO) when we trained ViTDet, ViT-B five times with varying random seeds.

The above models were trained and measured on 8-node with 64 NVIDIA A100 GPUs in total. *: Activation checkpointing is used.


## Training
All configs can be trained with:

```
../../tools/lazyconfig_train_net.py --config-file configs/path/to/config.py
```
By default, we use 64 GPUs with batch size as 64 for training.

## Evaluation
Model evaluation can be done similarly:
```
../../tools/lazyconfig_train_net.py --config-file configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


## <a name="CitingViTDet"></a>Citing ViTDet

If you use ViTDet, please use the following BibTeX entry.

```BibTeX
@article{li2022exploring,
  title={Exploring plain vision transformer backbones for object detection},
  author={Li, Yanghao and Mao, Hanzi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2203.16527},
  year={2022}
}
```
