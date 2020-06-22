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

# Quick Start

See [ Getting Started ](doc/GETTING_STARTED.md)

# Model Zoo and Baselines

We provide a number of baseline results and trained models available for download. See [Model Zoo](doc/MODEL_ZOO.md) for details.

# License

Detectron2 is released under the [Apache 2.0 license](../../LICENSE)

## <a name="CitingDensePose"></a>Citing DensePose

If you use DensePose, please take the references from the following BibTeX entries:

For DensePose with estimated confidences:

```
@InProceedings{Neverova2019DensePoseConfidences,
    title = {Correlated Uncertainty for Learning Dense Correspondences from Noisy Labels},
    author = {Neverova, Natalia and Novotny, David and Vedaldi, Andrea},
    journal = {Advances in Neural Information Processing Systems},
    year = {2019},
}
```

For the original DensePose:

```
@InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

