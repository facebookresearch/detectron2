# Pointly-Supervised Instance Segmentation

Bowen Cheng, Omkar Parkhi, Alexander Kirillov

[[`arXiv`](https://arxiv.org/abs/2104.06404)] [[`Project`](https://bowenc0221.github.io/point-sup)] [[`BibTeX`](#CitingPointSup)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/cheng2021pointly.png" width="50%" height="50%"/>
</div><br/>

## Data preparation
Please follow these steps to prepare your datasets:
1. Follow official Detectron2 instruction to prepare COCO dataset. Set up `DETECTRON2_DATASETS` environment variable to the location of your Detectron2 dataset.
2. Generate 10-points annotations for COCO by running: `python tools/prepare_coco_point_annotations_without_masks.py 10`

## Training

To train a model with 8 GPUs run:
```bash
python train_net.py --config-file configs/mask_rcnn_R_50_FPN_3x_point_sup_point_aug_coco.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/mask_rcnn_R_50_FPN_3x_point_sup_point_aug_coco.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## <a name="CitingPointSup"></a>Citing Pointly-Supervised Instance Segmentation

If you use PointSup, please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2022pointly,
  title={Pointly-Supervised Instance Segmentation},
  author={Cheng, Bowen and Parkhi, Omkar and Kirillov, Alexander},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2617--2626},
  year={2022}
}
```
