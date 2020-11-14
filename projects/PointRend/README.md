# PointRend: Image Segmentation as Rendering

Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick

[[`arXiv`](https://arxiv.org/abs/1912.08193)] [[`BibTeX`](#CitingPointRend)]

<div align="center">
  <img src="https://alexander-kirillov.github.io/images/kirillov2019pointrend.jpg"/>
</div><br/>

In this repository, we release code for PointRend in Detectron2. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models.

## Quick start and visualization

This [Colab Notebook](https://colab.research.google.com/drive/1isGPL5h5_cKoPPhVL9XhMokRtHDvmMVL) tutorial contains examples of PointRend usage and visualizations of its point sampling stages.

## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/PointRend
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/PointRend
python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

# Pretrained Models

## Instance Segmentation
#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Mask<br/>head</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">mask<br/>AP&ast;</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml">PointRend</a></td>
<td align="center">R50-FPN</td>
<td align="center">1&times;</td>
<td align="center">224&times;224</td>
<td align="center">36.2</td>
<td align="center">39.7</td>
<td align="center">164254221</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/model_final_736f5a.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/metrics.json">metrics</a></td>
</tr>
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml">PointRend</a></td>
<td align="center">R50-FPN</td>
<td align="center">3&times;</td>
<td align="center">224&times;224</td>
<td align="center">38.3</td>
<td align="center">41.6</td>
<td align="center">164955410</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/metrics.json">metrics</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml">PointRend</a></td>
<td align="center">R101-FPN</td>
<td align="center">3&times;</td>
<td align="center">224&times;224</td>
<td align="center">40.1</td>
<td align="center">43.8</td>
<td align="center"></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco/28119983/model_final_3f4d2a.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco/28119983/metrics.json">metrics</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml">PointRend</a></td>
<td align="center">X101-FPN</td>
<td align="center">3&times;</td>
<td align="center">224&times;224</td>
<td align="center">41.1</td>
<td align="center">44.7</td>
<td align="center"></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/metrics.json">metrics</a></td>
</tr>
</tbody></table>

AP&ast; is COCO mask AP evaluated against the higher-quality LVIS annotations; see the paper for details.
Run `python detectron2/datasets/prepare_cocofied_lvis.py` to prepare GT files for AP&ast; evaluation.
Since LVIS annotations are not exhaustive, `lvis-api` and not `cocoapi` should be used to evaluate AP&ast;.

#### Cityscapes
Cityscapes model is trained with ImageNet pretraining.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Mask<br/>head</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes.yaml">PointRend</a></td>
<td align="center">R50-FPN</td>
<td align="center">1&times;</td>
<td align="center">224&times;224</td>
<td align="center">35.9</td>
<td align="center">164255101</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes/164255101/model_final_115bfb.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes/164255101/metrics.json">metrics</a></td>
</tr>
</tbody></table>


## Semantic Segmentation

#### Cityscapes
Cityscapes model is trained with ImageNet pretraining.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml">SemanticFPN + PointRend</a></td>
<td align="center">R101-FPN</td>
<td align="center">1024&times;2048</td>
<td align="center">78.9</td>
<td align="center">202576688</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/metrics.json">metrics</a></td>
</tr>
</tbody></table>

## <a name="CitingPointRend"></a>Citing PointRend

If you use PointRend, please use the following BibTeX entry.

```BibTeX
@InProceedings{kirillov2019pointrend,
  title={{PointRend}: Image Segmentation as Rendering},
  author={Alexander Kirillov and Yuxin Wu and Kaiming He and Ross Girshick},
  journal={ArXiv:1912.08193},
  year={2019}
}
```
