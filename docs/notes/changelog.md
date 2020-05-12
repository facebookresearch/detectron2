# Change Log

### Releases
See release log at
[https://github.com/facebookresearch/detectron2/releases](https://github.com/facebookresearch/detectron2/releases).

### Notable Backward Incompatible Changes:

* 03/30/2020: Custom box head's `output_size` changed to `output_shape`.
* 02/14/2020,02/18/2020: Mask head and keypoint head now include logic for losses & inference. Custom heads
	should overwrite the feature computation by `layers()` method.
* 11/11/2019: `detectron2.data.detection_utils.read_image` transposes images with exif information.

### Config Version Change Log

* v1: Rename `RPN_HEAD.NAME` to `RPN.HEAD_NAME`.
* v2: A batch of rename of many configurations before release.

### Silent Regression in Historical Versions:

We list a few silent regressions since they may silently produce incorrect results and will be hard to debug.

* 04/01/2020 - 05/11/2020: Bad accuracy if `TRAIN_ON_PRED_BOXES` is set to True.
* 03/30/2020 - 04/01/2020: ResNets are not correctly built.
* 12/19/2019 - 12/26/2019: Using aspect ratio grouping causes a drop in accuracy.
* release - 11/9/2019: Test time augmentation does not predict the last category.
