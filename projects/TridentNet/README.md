
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
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --num_gpus 8
```

## Evaluation

Model evaluationcan be done similarly:
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml --eval-only MODEL.WEIGHTS model.pth
```

## Results on MS-COCO in Detectron2

|Model|Backbone|Head|lr sched|AP|AP50|AP75|APs|APm|APl|
|-----|--------|----|--------------|--|----|----|---|---|---|
|Faster|R50-C4|C5-512ROI|1X|35.7|56.1|38.0|19.2|40.9|48.7|
|TridentFast|R50-C4|C5-128ROI|1X|37.9|57.8|40.7|19.7|42.1|54.2|
|Faster|R50-C4|C5-512ROI|3X|38.4|58.7|41.3|20.7|42.7|53.1|
|TridentFast|R50-C4|C5-128ROI|3X|41.0|60.9|44.2|22.7|45.2|57.0|
|Faster|R101-C4|C5-512ROI|3X|41.1|61.4|44.0|22.2|45.5|55.9|
|TridentFast|R101-C4|C5-128ROI|3X|43.4|62.9|46.6|24.2|47.9|59.9|


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

