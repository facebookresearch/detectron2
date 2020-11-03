# Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

Bowen Cheng, Maxwell D. Collins, Yukun Zhu, Ting Liu, Thomas S. Huang, Hartwig Adam, Liang-Chieh Chen

[[`arXiv`](https://arxiv.org/abs/1911.10194)] [[`BibTeX`](#CitingPanopticDeepLab)] [[`Reference implementation`](https://github.com/bowenc0221/panoptic-deeplab)]

<div align="center">
  <img src="https://github.com/bowenc0221/panoptic-deeplab/blob/master/docs/panoptic_deeplab.png"/>
</div><br/>

## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).
To use cityscapes, prepare data follow the [tutorial](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-cityscapes).

## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/Panoptic-DeepLab
python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## Cityscapes Panoptic Segmentation
Cityscapes models are trained with ImageNet pretraining.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">AP</th>
<th valign="bottom">Memory (M)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left">Panoptic-DeepLab</td>
<td align="center">R50-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 58.6 </td>
<td align="center"> 80.9 </td>
<td align="center"> 71.2 </td>
<td align="center"> 75.9 </td>
<td align="center"> 29.8 </td>
<td align="center"> 8668 </td>
<td align="center"> - </td>
<td align="center">model&nbsp;|&nbsp;metrics</td>
</tr>
 <tr><td align="left"><a href="config/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">R52-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 60.3 </td>
<td align="center"> 81.5 </td>
<td align="center"> 72.9 </td>
<td align="center"> 78.2 </td>
<td align="center"> 33.2 </td>
<td align="center"> 9682 </td>
<td align="center">  </td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/metrics.json
">metrics</a></td>
</tr>
</tbody></table>

Note:
- [R52](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-52.pkl): a ResNet-50 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers. We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
- DC5 means using dilated convolution in `res5`.
- We use a smaller training crop size (512x1024) than the original paper (1025x2049), we find using larger crop size (1024x2048) could further improve PQ by 1.5% but also degrades AP by 3%.
- This implementation currently uses a much heavier head than the original paper.
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
  outputs now takes more time than the network itself.

## <a name="CitingPanopticDeepLab"></a>Citing Panoptic-DeepLab

If you use Panoptic-DeepLab, please use the following BibTeX entry.

*   CVPR 2020 paper:

```
@inproceedings{cheng2020panoptic,
  title={Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2020}
}
```

*   ICCV 2019 COCO-Mapillary workshp challenge report:

```
@inproceedings{cheng2019panoptic,
  title={Panoptic-DeepLab},
  author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
  booktitle={ICCV COCO + Mapillary Joint Recognition Challenge Workshop},
  year={2019}
}
```
