# Change Log and Backward Compatibility

### Releases
See release logs at
[https://github.com/facebookresearch/detectron2/releases](https://github.com/facebookresearch/detectron2/releases)
for new updates.

### Backward Compatibility

Due to the research nature of what the library does, there might be backward incompatible changes.
But we try to reduce users' disruption by the following ways:
* APIs listed in [API documentation](https://detectron2.readthedocs.io/modules/index.html), including
  function/class names, their arguments, and documented class attributes, are considered *stable* unless
  otherwise noted in the documentation.
  They are less likely to be broken, but if needed, will trigger a deprecation warning for a reasonable period
  before getting broken, and will be documented in release logs.
* Others functions/classses/attributes are considered internal, and are more likely to change.
  However, we're aware that some of them may be already used by other projects, and in particular we may
  use them for convenience among projects under `detectron2/projects`.
  For such APIs, we may treat them as stable APIs and also apply the above strategies.
  They may be promoted to stable when we're ready.
* Projects under "detectron2/projects" or imported with "detectron2.projects" are research projects
  and are all considered experimental.
* Classes/functions that contain the word "default" or are explicitly documented to produce
  "default behavior" may change their behaviors when new features are added.

Despite of the possible breakage, if a third-party project would like to keep up with the latest updates
in detectron2, using it as a library will still be less disruptive than forking, because
the frequency and scope of API changes will be much smaller than code changes.

To see such changes, search for "incompatible changes" in [release logs](https://github.com/facebookresearch/detectron2/releases).

### Config Version Change Log

Detectron2's config version has not been changed since open source.
There is no need for an open source user to worry about this.

* v1: Rename `RPN_HEAD.NAME` to `RPN.HEAD_NAME`.
* v2: A batch of rename of many configurations before release.

### Silent Regressions in Historical Versions:

We list a few silent regressions, since they may silently produce incorrect results and will be hard to debug.

* 04/01/2020 - 05/11/2020: Bad accuracy if `TRAIN_ON_PRED_BOXES` is set to True.
* 03/30/2020 - 04/01/2020: ResNets are not correctly built.
* 12/19/2019 - 12/26/2019: Using aspect ratio grouping causes a drop in accuracy.
* - 11/9/2019: Test time augmentation does not predict the last category.
