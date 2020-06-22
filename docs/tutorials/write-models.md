# Write Models

If you are trying to do something completely new, you may wish to implement
a model entirely from scratch within detectron2. However, in many situations you may
be interested in modifying or extending some components of an existing model.
Therefore, we also provide a registration mechanism that lets you override the
behavior of certain internal components of standard models.

For example, to add a new backbone, import this code in your code:
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackBone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```
Then, you can use `cfg.MODEL.BACKBONE.NAME = 'ToyBackBone'` in your config object.
`build_model(cfg)` will then call your `ToyBackBone` instead.

As another example, to add new abilities to the ROI heads in the Generalized R-CNN meta-architecture,
you can implement a new
[ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads) subclass and put it in the `ROI_HEADS_REGISTRY`.
See [densepose in detectron2](../../projects/DensePose)
and [meshrcnn](https://github.com/facebookresearch/meshrcnn)
for examples that implement new ROIHeads to perform new tasks.
And [projects/](../../projects/)
contains more examples that implement different architectures.

A complete list of registries can be found in [API documentation](../modules/modeling.html#model-registries).
You can register components in these registries to customize different parts of a model, or the
entire model.
