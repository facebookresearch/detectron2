# DensePose CSE with Cycle Losses

This release follows the paper [Neverova et al, 2021]() and
adds CSE datasets with more annotations, better CSE animal models
to the model zoo, losses to ensure cycle consistency for models and mesh
alignment evaluator. In particular:

* [Pixel to shape](../densepose/modeling/losses/cycle_pix2shape.py) and [shape to shape](../densepose/modeling/losses/cycle_shape2shape.py) cycle consistency losses;
* Mesh alignment [evaluator](../densepose/evaluation/mesh_alignment_evaluator.py);
* Existing CSE datasets renamed to [ds1_train](https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds1_train_v1.json) and [ds1_val](https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds1_val_v1.json);
* New CSE datasets [ds2_train](https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds2_train_v1.json) and [ds2_val](https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds2_val_v1.json) added;
* Better CSE animal models trained with the 16k schedule added to the [model zoo](DENSEPOSE_CSE.md#animal-cse-models).
