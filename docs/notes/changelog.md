# Change Log

### Releases
See release log at
[https://github.com/facebookresearch/detectron2/releases](https://github.com/facebookresearch/detectron2/releases)

### Notable Backward Incompatible Changes:

* 02/14/2020,02/18/2020: Mask head and keypoint head now include logic for losses & inference. Custom heads
	should overwrite the feature computation by `layers()` method.
* 11/11/2019: `detectron2.data.detection_utils.read_image` transposes images with exif information.

### Config Version Change Log

* v1: Rename `RPN_HEAD.NAME` to `RPN.HEAD_NAME`.
* v2: A batch of rename of many configurations before release.

### Known Bugs in Historical Versions:
* Dec 19 - Dec 26: Using aspect ratio grouping causes a drop in accuracy.
* Oct 10 - Nov 9: Test time augmentation does not predict the last category.
