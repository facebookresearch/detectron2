
# DensePose in Detectron2
**Dense Human Pose Estimation In The Wild**

_Rıza Alp Güler, Natalia Neverova, Iasonas Kokkinos_

[[`densepose.org`](https://densepose.org)] [[`arXiv`](https://arxiv.org/abs/1802.00434)] [[`BibTeX`](#CitingDensePose)]

Dense human pose estimation aims at mapping all human pixels of an RGB image to the 3D surface of the human body.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1qfSOkpueo1kVZbXOuQJJhyagKjMgepsz" width="700px" />
</div>

In this repository, we provide the code to train and evaluate DensePose-RCNN. We also provide tools to visualize
DensePose annotation and results.

## Training

To train a model one can call
```bash
python /path/to/detectron2/projects/DensePose/train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end DensePose-RCNN training with ResNet-50 FPN backbone on a single GPU,
one should execute:
```bash
python /path/to/detectron2/projects/DensePose/train_net.py --config-file configs/densepose_R_50_FPN_s1x.yaml
```

## Evaluation

Model evaluation can be done in the same way as training, except for an additional flag `--eval-only` and
model location specification through `MODEL.WEIGHTS model.pth` in the command line
```bash
python /path/to/detectron2/projects/DensePose/train_net.py --config-file configs/densepose_R_50_FPN_s1x.yaml --eval-only MODEL.WEIGHTS model.pth
```

## Tools

We provide tools which allow one to:
 - easily view DensePose annotated data in a dataset;
 - perform DensePose inference on a set of images;
 - visualize DensePose model results;

`query_db` is a tool to print or visualize DensePose data in a dataset.
Details on this tool can be found in [`TOOL_QUERY_DB.md`](doc/TOOL_QUERY_DB.md)

`apply_net` is a tool to print or visualize DensePose results.
Details on this tool can be found in [`TOOL_APPLY_NET.md`](doc/TOOL_APPLY_NET.md)

## <a name="CitingDensePose"></a>Citing DensePose

If you use DensePose, please use the following BibTeX entry.

```
@InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

