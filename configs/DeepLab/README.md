# DeepLab in Detectron2

In this repository, we implement DeepLabV3 and DeepLabV3+ in Detectron2.

## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/DeepLab
python train_net.py --config-file configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/DeepLab
python train_net.py --config-file configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## Cityscapes Semantic Segmentation
Cityscapes models are trained with ImageNet pretraining.

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
 <tr><td align="left">DeepLabV3</td>
<td align="center">R101-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 76.7 </td>
<td align="center"> - </td>
<td align="center"> - &nbsp;|&nbsp; - </td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-SemanticSegmentation/deeplab_v3_R_103_os16_mg124_poly_90k_bs16.yaml">DeepLabV3</a></td>
<td align="center">R103-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 78.5 </td>
<td align="center"> 28041665 </td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/DeepLab/Cityscapes-SemanticSegmentation/deeplab_v3_R_103_os16_mg124_poly_90k_bs16/28041665/model_final_0dff1b.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/DeepLab/Cityscapes-SemanticSegmentation/deeplab_v3_R_103_os16_mg124_poly_90k_bs16/28041665/metrics.json
">metrics</a></td>
</tr>
 <tr><td align="left">DeepLabV3+</td>
<td align="center">R101-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 78.1 </td>
<td align="center"> - </td>
<td align="center"> - &nbsp;|&nbsp; - </td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml">DeepLabV3+</a></td>
<td align="center">R103-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 80.0 </td>
<td align="center">28054032</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/DeepLab/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16/28054032/model_final_a8a355.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/DeepLab/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16/28054032/metrics.json
">metrics</a></td>
</tr>
</tbody></table>

Note:
- [R103](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. 
This modification has been used in most semantic segmentation papers. We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
- DC5 means using dilated convolution in `res5`.

## <a name="CitingDeepLab"></a>Citing DeepLab

If you use DeepLab, please use the following BibTeX entry.

*   DeepLabv3+:

```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

*   DeepLabv3:

```
@article{deeplabv32018,
  title={Rethinking atrous convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={arXiv:1706.05587},
  year={2017}
}
```
