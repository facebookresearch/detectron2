# Write Models

If you are trying to do something completely new, you may wish to implement
a model entirely from scratch. However, in many situations you may
be interested in modifying or extending some components of an existing model.
Therefore, we also provide mechanisms that let users override the
behavior of certain internal components of standard models.


## Register New Components

For common concepts that users often want to customize, such as "backbone feature extractor", "box head",
we provide a registration mechanism for users to inject custom implementation that
will be immediately available to use in config files.

For example, to add a new backbone, import this code in your code:
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```

In this code, we implement a new backbone following the interface of the
[Backbone](../modules/modeling.html#detectron2.modeling.Backbone) class,
and register it into the [BACKBONE_REGISTRY](../modules/modeling.html#detectron2.modeling.BACKBONE_REGISTRY)
which requires subclasses of `Backbone`.
After importing this code, detectron2 can link the name of the class to its implementation. Therefore you can write the following code:

```python
cfg = ...   # read a config
cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # or set it in the config file
model = build_model(cfg)  # it will find `ToyBackbone` defined above
```

As another example, to add new abilities to the ROI heads in the Generalized R-CNN meta-architecture,
you can implement a new
[ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads) subclass and put it in the `ROI_HEADS_REGISTRY`.
[DensePose](../../projects/DensePose)
and [MeshRCNN](https://github.com/facebookresearch/meshrcnn)
are two examples that implement new ROIHeads to perform new tasks.
And [projects/](../../projects/)
contains more examples that implement different architectures.

A complete list of registries can be found in [API documentation](../modules/modeling.html#model-registries).
You can register components in these registries to customize different parts of a model, or the
entire model.

## Construct Models with Explicit Arguments

Registry is a bridge to connect names in config files to the actual code.
They are meant to cover a few main components that users frequently need to replace.
However, the capability of a text-based config file is sometimes limited and
some deeper customization may be available only through writing code.

Most model components in detectron2 have a clear `__init__` interface that documents
what input arguments it needs. Calling them with custom arguments will give you a custom variant
of the model.

As an example, to use __custom loss function__ in the box head of a Faster R-CNN, we can do the following:

1. Losses are currently computed in [FastRCNNOutputLayers](../modules/modeling.html#detectron2.modeling.FastRCNNOutputLayers).
   We need to implement a variant or a subclass of it, with custom loss functions, named  `MyRCNNOutput`.
2. Call `StandardROIHeads` with `box_predictor=MyRCNNOutput()` argument instead of the builtin `FastRCNNOutputLayers`.
   If all other arguments should stay unchanged, this can be easily achieved by using the [configurable `__init__`](../modules/config.html#detectron2.config.configurable) mechanism:

   ```python
   roi_heads = StandardROIHeads(
     cfg, backbone.output_shape(),
     box_predictor=MyRCNNOutput(...)
   )
   ```
3. (optional) If we want to enable this new model from a config file, registration is needed:
   ```python
   @ROI_HEADS_REGISTRY.register()
   class MyStandardROIHeads(StandardROIHeads):
     def __init__(self, cfg, input_shape):
       super().__init__(cfg, input_shape,
                        box_predictor=MyRCNNOutput(...))
   ```
