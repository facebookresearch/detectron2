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

Note: Unlike the system-level comparisons in the paper, these models use a lower resolution (1024 instead of 1280) and standard NMS (instead of soft NMS). As a result, they have slightly lower box and mask AP.

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
