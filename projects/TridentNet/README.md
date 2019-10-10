
# TridentNet in Detectron2
**Scale-Aware Trident Networks for Object Detection**

Yanghao Li\*, Yuntao Chen\*, Naiyan Wang, Zhaoxiang Zhang

[[`arXiv`](https://arxiv.org/abs/1802.00434)] [[`BibTeX`](#CitingTridentNet)]

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=10THEPdIPmf3ooMyNzrfZbpWihEBvixwt" width="700px" />
</div>

In this repository, we implement TridentNet-Fast in the Detectron2 framework. Trident Network (TridentNet) aims to generate scale-specific feature maps with a uniform representational power. We construct a parallel multi-branch architecture in which each branch shares the same transformation parameters but with different receptive fields. TridentNet-Fast is a fast approximation version of TridentNet that could achieve significant improvements without any additional parameters and computational cost.

## Training

To train a model one can call
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end TridentNet training with ResNet-50 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file /path/to/detectron2/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml --num_gpus 8
```

## Testing

Model testing can be done in the same way as training, except for an additional flag `--eval-only` and
model location specification through `MODEL.WEIGHT model.pth` in the command line
```bash
python /path/to/detectron2/projects/TridentNet/train_net.py --config-file /path/to/detectron2/projects/TridentNet/configs/tridentnet_fast_R_50_C4_1x.yaml --eval-only MODEL.WEIGHT model.pth
```

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

