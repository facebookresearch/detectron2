# Using and Writing Models

Models (and their sub-models) in detectron2 are built by
functions such as `build_model`, `build_backbone`, `build_roi_heads`:
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

In some cases, e.g. if you are trying to do something completely new, you may wish to implement
a model entirely from scratch within detectron2. However, in many situations you may
be interested in modifying or extending some components of an existing model.
Therefore, we also provide a registration mechanism that lets you override the
behavior of certain internal components of standard models.

For example, to add a new backbone, import this code:
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
@BACKBONE_REGISTRY.register()
class NewBackBone(Backbone):
  def __init__(self, cfg, input_shape):
    # create your own backbone
```
which will allow you to use `cfg.MODEL.BACKBONE.NAME = 'NewBackBone'` in your config file.

As another example, to add new abilities to the ROI heads in the Generalized R-CNN meta-architecture,
you can implement a new
[ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads) subclass and put it in the `ROI_HEADS_REGISTRY`.
See [densepose in detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose)
for an example.

Other registries can be found in [API documentation](../modules/modeling.html).
