# Use Models

Models (and their sub-models) in detectron2 are built by
functions such as `build_model`, `build_backbone`, `build_roi_heads`:
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

Note that `build_model` only builds the model structure, and fill it with random parameters.
To load an existing checkpoint to the model, use
`DetectionCheckpointer(model).load(file_path)`.
Detectron2 recognizes models in pytorch's `.pth` format, as well as the `.pkl` files
in our model zoo.

You can use a model by just `outputs = model(inputs)`.
Next, we explain the inputs/outputs format used by the builtin models in detectron2.

[DefaultPredictor](../modules/engine.html#detectron2.engine.defaults.DefaultPredictor)
is a wrapper around model that provides the default behavior for regular inference. It includes model loading as
well as preprocessing, and operates on single image rather than batches.


### Model Input Format

All builtin models take a `list[dict]` as the inputs. Each dict
corresponds to information about one image.

The dict may contain the following keys:

* "image": `Tensor` in (C, H, W) format. The meaning of channels are defined by `cfg.INPUT.FORMAT`.
* "instances": an [Instances](../modules/structures.html#detectron2.structures.Instances)
  object, with the following fields:
  + "gt_boxes": a [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing N boxes, one for each instance.
  + "gt_classes": `Tensor` of long type, a vector of N labels, in range [0, num_categories).
  + "gt_masks": a [PolygonMasks](../modules/structures.html#detectron2.structures.PolygonMasks)
    or [BitMasks](../modules/structures.html#detectron2.structures.BitMasks) object storing N masks, one for each instance.
  + "gt_keypoints": a [Keypoints](../modules/structures.html#detectron2.structures.Keypoints)
    object storing N keypoint sets, one for each instance.
* "proposals": an [Instances](../modules/structures.html#detectron2.structures.Instances)
  object used only in Fast R-CNN style models, with the following fields:
  + "proposal_boxes": a [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing P proposal boxes.
  + "objectness_logits": `Tensor`, a vector of P scores, one for each proposal.
* "height", "width": the **desired** output height and width, which is not necessarily the same
  as the height or width of the `image` input field.
  For example, the `image` input field might be a resized image,
  but you may want the outputs to be in **original** resolution.

  If provided, the model will produce output in this resolution,
  rather than in the resolution of the `image` as input into the model. This is more efficient and accurate.
* "sem_seg": `Tensor[int]` in (H, W) format. The semantic segmentation ground truth.
  Values represent category labels starting from 0.


#### How it connects to data loader:

The output of the default [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper) is a dict
that follows the above format.
After the data loader performs batching, it becomes `list[dict]` which the builtin models support.


### Model Output Format

When in training mode, the builtin models output a `dict[str->ScalarTensor]` with all the losses.

When in inference mode, the builtin models output a `list[dict]`, one dict for each image.
Based on the tasks the model is doing, each dict may contain the following fields:

* "instances": [Instances](../modules/structures.html#detectron2.structures.Instances)
  object with the following fields:
  * "pred_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing N boxes, one for each detected instance.
  * "scores": `Tensor`, a vector of N scores.
  * "pred_classes": `Tensor`, a vector of N labels in range [0, num_categories).
  + "pred_masks": a `Tensor` of shape (N, H, W), masks for each detected instance.
  + "pred_keypoints": a `Tensor` of shape (N, num_keypoint, 3).
    Each row in the last dimension is (x, y, score). Scores are larger than 0.
* "sem_seg": `Tensor` of (num_categories, H, W), the semantic segmentation prediction.
* "proposals": [Instances](../modules/structures.html#detectron2.structures.Instances)
  object with the following fields:
  * "proposal_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes)
    object storing N boxes.
  * "objectness_logits": a torch vector of N scores.
* "panoptic_seg": A tuple of `(Tensor, list[dict])`. The tensor has shape (H, W), where each element
  represent the segment id of the pixel. Each dict describes one segment id and has the following fields:
  * "id": the segment id
  * "isthing": whether the segment is a thing or stuff
  * "category_id": the category id of this segment. It represents the thing
       class id when `isthing==True`, and the stuff class id otherwise.


### How to use a model in your code:

Contruct your own `list[dict]` as inputs, with the necessary keys. Then call `outputs = model(inputs)`.
For example, in order to do inference, provide dicts with "image", and optionally "height" and "width".

Note that when in training mode, all models are required to be used under an `EventStorage`.
The training statistics will be put into the storage:
```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
  losses = model(inputs)
```

Another small thing to remember: detectron2 models do not support `model.to(device)` or `model.cpu()`.
The device is defined in `cfg.MODEL.DEVICE` and cannot be changed afterwards.


### Partially execute a model:

Sometimes you may want to obtain an intermediate tensor inside a model.
Since there are typically hundreds of intermediate tensors, there isn't an API that provides you
the intermediate result you need.
You have the following options:

1. Write a (sub)model. Following the [tutorial](write-models.html), you can
   rewrite a model component (e.g. a head of a model), such that it
   does the same thing as the existing component, but returns the output
   you need.
2. Partially execute a model. You can create the model as usual,
   but use custom code to execute it instead of its `forward()`. For example,
   the following code obtains mask features before mask head.

```python
images = ImageList(...)  # preprocessed input tensor
model = build_model(cfg)
features = model.backbone(images.tensor)
proposals, _ = model.proposal_generator(images, features)
instances = model.roi_heads._forward_box(
  [features[k] for k in model.roi_heads.in_features],
  proposals
)
mask_features = model.roi_heads.mask_pooler(features, [x.pred_boxes for x in instances])
```

Note that both options require you to read the existing forward code to understand
how to write code to obtain the outputs you need.
