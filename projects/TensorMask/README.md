
# TensorMask in Detectron2
**A Foundation for Dense Object Segmentation**

Xinlei Chen, Ross Girshick, Kaiming He, Piotr Dollár

[[`arXiv`](https://arxiv.org/abs/1903.12174)] [[`BibTeX`](#CitingTensorMask)]

<div align="center">
  <img src="http://xinleic.xyz/images/tmask.png" width="700px" />
</div>

In this repository, we release code for TensorMask in Detectron2.
TensorMask is a dense sliding-window instance segmentation framework that, for the first time, achieves results close to the well-developed Mask R-CNN framework -- both qualitatively and quantitatively. It establishes a conceptually complementary direction for object instance segmentation research. 

## Installation
To install, first setup Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Then to compile the TensorMask-specific op (`swap_align2nat`):
```bash
cd /path/to/detectron2/projects/TensorMask
python setup.py build develop
```

## Training

To train a model, run:
```bash
python /path/to/detectron2/projects/TensorMask/train_net.py --config-file <config.yaml>
```

For example, to launch TensorMask BiPyramid training (1x schedule) with ResNet-50 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/detectron2/projects/TensorMask/train_net.py --config-file configs/tensormask_R_50_FPN_1x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly (6x schedule with scale augmentation):
```bash
python /path/to/detectron2/projects/TensorMask/train_net.py --config-file configs/tensormask_R_50_FPN_6x.yaml --eval-only MODEL.WEIGHTS model.pth
```

# Model Zoo and Baselines

(coming soon)


## <a name="CitingTensorMask"></a>Citing TensorMask

If you use TensorMask, please use the following BibTeX entry.

```
@InProceedings{chen2019tensormask,
  title={Tensormask: A Foundation for Dense Object Segmentation},
  author={Chen, Xinlei and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

