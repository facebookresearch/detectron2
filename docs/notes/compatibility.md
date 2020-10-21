# Compatibility with Other Libraries

## Compatibility with Detectron (and maskrcnn-benchmark)

Detectron2 addresses some legacy issues left in Detectron. As a result, their models
are not compatible:
running inference with the same model weights will produce different results in the two code bases.

The major differences regarding inference are:

- The height and width of a box with corners (x1, y1) and (x2, y2) is now computed more naturally as
  width = x2 - x1 and height = y2 - y1;
  In Detectron, a "+ 1" was added both height and width.

  Note that the relevant ops in Caffe2 have [adopted this change of convention](https://github.com/pytorch/pytorch/pull/20550)
  with an extra option.
  So it is still possible to run inference with a Detectron2-trained model in Caffe2.

  The change in height/width calculations most notably changes:
  - encoding/decoding in bounding box regression.
  - non-maximum suppression. The effect here is very negligible, though.

- RPN now uses simpler anchors with fewer quantization artifacts.

  In Detectron, the anchors were quantized and
  [do not have accurate areas](https://github.com/facebookresearch/Detectron/issues/227).
  In Detectron2, the anchors are center-aligned to feature grid points and not quantized.

- Classification layers have a different ordering of class labels.

  This involves any trainable parameter with shape (..., num_categories + 1, ...).
  In Detectron2, integer labels [0, K-1] correspond to the K = num_categories object categories
  and the label "K" corresponds to the special "background" category.
  In Detectron, label "0" means background, and labels [1, K] correspond to the K categories.

- ROIAlign is implemented differently. The new implementation is [available in Caffe2](https://github.com/pytorch/pytorch/pull/23706).

  1. All the ROIs are shifted by half a pixel compared to Detectron in order to create better image-feature-map alignment.
     See `layers/roi_align.py` for details.
     To enable the old behavior, use `ROIAlign(aligned=False)`, or `POOLER_TYPE=ROIAlign` instead of
     `ROIAlignV2` (the default).

  1. The ROIs are not required to have a minimum size of 1.
     This will lead to tiny differences in the output, but should be negligible.

- Mask inference function is different.

  In Detectron2, the "paste_mask" function is different and should be more accurate than in Detectron. This change
  can improve mask AP on COCO by ~0.5% absolute.

There are some other differences in training as well, but they won't affect
model-level compatibility. The major ones are:

- We fixed a [bug](https://github.com/facebookresearch/Detectron/issues/459) in
  Detectron, by making `RPN.POST_NMS_TOPK_TRAIN` per-image, rather than per-batch.
  The fix may lead to a small accuracy drop for a few models (e.g. keypoint
  detection) and will require some parameter tuning to match the Detectron results.
- For simplicity, we change the default loss in bounding box regression to L1 loss, instead of smooth L1 loss.
  We have observed that this tends to slightly decrease box AP50 while improving box AP for higher
  overlap thresholds (and leading to a slight overall improvement in box AP).
- We interpret the coordinates in COCO bounding box and segmentation annotations
  as coordinates in range `[0, width]` or `[0, height]`. The coordinates in
  COCO keypoint annotations are interpreted as pixel indices in range `[0, width - 1]` or `[0, height - 1]`.
  Note that this affects how flip augmentation is implemented.


We will later share more details and rationale behind the above mentioned issues
about pixels, coordinates, and "+1"s.


## Compatibility with Caffe2

As mentioned above, despite the incompatibilities with Detectron, the relevant
ops have been implemented in Caffe2.
Therefore, models trained with detectron2 can be converted in Caffe2.
See [Deployment](../tutorials/deployment.md) for the tutorial.

## Compatibility with TensorFlow

Most ops are available in TensorFlow, although some tiny differences in
the implementation of resize / ROIAlign / padding need to be addressed.
A working conversion script is provided by [tensorpack Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/convert_d2)
to run a standard detectron2 model in TensorFlow.
