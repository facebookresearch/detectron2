# MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer*

[[`arXiv`](https://arxiv.org/abs/2112.01526)] [[`BibTeX`](#CitingMViTv2)]

In this repository, we provide detection configs and models for MViTv2 (CVPR 2022) in Detectron2. For image classification tasks, please refer to [MViTv2 repo](https://github.com/facebookresearch/mvit).

## Results and Pretrained Models

### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">Method</th>
<th valign="bottom">epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">#params</th>
<th valign="bottom">FLOPS</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_mvitv2_t_3x -->
 <tr><td align="left"><a href="configs/mask_rcnn_mvitv2_t_3x.py">MViTV2-T</a></td>
<td align="center">IN1K</td>
<td align="center">Mask R-CNN</td>
<td align="center">36</td>
<td align="center">48.3</td>
<td align="center">43.8</td>
<td align="center">44M</td>
<td align="center">279G</td>
<td align="center">307611773</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/mask_rcnn_mvitv2_t_3x/f307611773/model_final_1a1c30.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_t_3x -->
 <tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_t_3x.py">MViTV2-T</a></td>
<td align="center">IN1K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">36</td>
<td align="center">52.2</td>
<td align="center">45.0</td>
<td align="center">76M</td>
<td align="center">701G</td>
<td align="center">308344828</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_t_3x/f308344828/model_final_c6967a.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_s_3x -->
<tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_s_3x.py">MViTV2-S</a></td>
<td align="center">IN1K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">36</td>
<td align="center">53.2</td>
<td align="center">46.0</td>
<td align="center">87M</td>
<td align="center">748G</td>
<td align="center">308344647</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_s_3x/f308344647/model_final_279baf.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_b_3x -->
<tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_b_3x.py">MViTV2-B</a></td>
<td align="center">IN1K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">36</td>
<td align="center">54.1</td>
<td align="center">46.7</td>
<td align="center">103M</td>
<td align="center">814G</td>
<td align="center">308109448</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_b_3x/f308109448/model_final_421a91.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_b_in21k_3x -->
<tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_b_in21k_3x.py">MViTV2-B</a></td>
<td align="center">IN21K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">36</td>
<td align="center">54.9</td>
<td align="center">47.4</td>
<td align="center">103M</td>
<td align="center">814G</td>
<td align="center">309003202</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_b_in12k_3x/f309003202/model_final_be5168.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_l_in21k_lsj_50ep -->
<tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_l_in21k_lsj_50ep.py">MViTV2-L</a></td>
<td align="center">IN21K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">50</td>
<td align="center">55.8</td>
<td align="center">48.3</td>
<td align="center">270M</td>
<td align="center">1519G</td>
<td align="center">308099658</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_l_in12k_lsj_50ep/f308099658/model_final_c41c5a.pkl">model</a></td>
</tr>
<!-- ROW: cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x -->
<tr><td align="left"><a href="configs/cascade_mask_rcnn_mvitv2_h_in21k_lsj_3x.py">MViTV2-H</a></td>
<td align="center">IN21K</td>
<td align="center">Cascade Mask R-CNN</td>
<td align="center">36</td>
<td align="center">56.1</td>
<td align="center">48.5</td>
<td align="center">718M</td>
<td align="center">3084G</td>
<td align="center">309013744</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_h_in12k_lsj_3x/f309013744/model_final_30d36b.pkl">model</a></td>
</tr>
</tbody></table>

Note that the above models were trained and measured on 8-node with 64 NVIDIA A100 GPUs in total. The ImageNet pre-trained model weights are obtained from [MViTv2 repo](https://github.com/facebookresearch/mvit).

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



## <a name="CitingMViTv2"></a>Citing MViTv2

If you use MViTv2, please use the following BibTeX entry.

```BibTeX
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```
