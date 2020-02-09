# PointRend: Image Segmentation as Rendering

Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick

[[`arXiv`](https://arxiv.org/abs/1912.08193)] [[`BibTeX`](#CitingPointRend)]

<div align="center">
  <img src="https://alexander-kirillov.github.io/images/kirillov2019pointrend.jpg"/>
</div><br/>

In this repository, we release code for PointRend in Detectron2. PointRend can be flexibly applied to both instance and semantic (**comming soon**) segmentation tasks by building on top of existing state-of-the-art models.

## Installation
Install Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). You are ready to go!

## Quick start and visualization

This [Colab Notebook](https://colab.research.google.com/drive/1isGPL5h5_cKoPPhVL9XhMokRtHDvmMVL) tutorial contains examples of PointRend usage and visualisations of its point sampling stages.

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
<td align="center">38.3</td>
<td align="center">164254221</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/model_final_88c6f8.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/metrics.json">metrics</a></td>
</tr>
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml">PointRend</a></td>
<td align="center">R50-FPN</td>
<td align="center">3&times;</td>
<td align="center">224&times;224</td>
<td align="center">38.3</td>
<td align="center">40.2</td>
<td align="center">164955410</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/metrics.json">metrics</a></td>
</tr>
</tbody></table>

AP&ast; is COCO mask AP evaluated against the higher-quality LVIS annotations; see the paper for details.

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
 <tr><td align="left"><a href="configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cs.yaml">PointRend</a></td>
<td align="center">R50-FPN</td>
<td align="center">1&times;</td>
<td align="center">224&times;224</td>
<td align="center">35.9</td>
<td align="center">164255101</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes/164255101/model_final_318a02.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes/164255101/metrics.json">metrics</a></td>
</tr>
</tbody></table>


## Semantic Segmentation

**[comming soon]**

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
