
# TridentNet in Detectron2
**Scale-Aware Trident Networks for Object Detection**

Yanghao Li\*, Yuntao Chen\*, Naiyan Wang, Zhaoxiang Zhang

[[`TridentNet`](https://github.com/TuSimple/simpledet/tree/master/models/tridentnet)] [[`arXiv`](https://arxiv.org/abs/1901.01892)] [[`BibTeX`](#CitingTridentNet)]

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=10THEPdIPmf3ooMyNzrfZbpWihEBvixwt" width="700px" />
</div>

In this repository, we implement TridentNet-Fast in Detectron2.
Trident Network (TridentNet) aims to generate scale-specific feature maps with a uniform representational power. We construct a parallel multi-branch architecture in which each branch shares the same transformation parameters but with different receptive fields. TridentNet-Fast is a fast approximation version of TridentNet that could achieve significant improvements without any additional parameters and computational cost.

## Training

To train a model, run
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end TridentNet training with ResNet-50 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS model.pth
```

## Results on MS-COCO in Detectron2

|Model|Backbone|Head|lr sched|AP|AP50|AP75|APs|APm|APl|download|
|-----|--------|----|--------|--|----|----|---|---|---|--------|
|Faster|R50-C4|C5-512ROI|1X|35.7|56.1|38.0|19.2|40.9|48.7|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a>|
|TridentFast|R50-C4|C5-128ROI|1X|38.0|58.1|40.8|19.5|42.2|54.6|<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_50_C4_1x/148572687/model_final_756cda.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_50_C4_1x/148572687/metrics.json">metrics</a>|
|Faster|R50-C4|C5-512ROI|3X|38.4|58.7|41.3|20.7|42.7|53.1|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/metrics.json">metrics</a>|
|TridentFast|R50-C4|C5-128ROI|3X|40.6|60.8|43.6|23.4|44.7|57.1|<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_50_C4_3x/148572287/model_final_e1027c.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_50_C4_3x/148572287/metrics.json">metrics</a>|
|Faster|R101-C4|C5-512ROI|3X|41.1|61.4|44.0|22.2|45.5|55.9|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/metrics.json">metrics</a>|
|TridentFast|R101-C4|C5-128ROI|3X|43.6|63.4|47.0|24.3|47.8|60.0|<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_101_C4_3x/148572198/model_final_164568.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/TridentNet/tridentnet_fast_R_101_C4_3x/148572198/metrics.json">metrics</a>|


## <a name="CitingTridentNet"></a>Citing TridentNet

If you use TridentNet, please use the following BibTeX entry.

```
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

