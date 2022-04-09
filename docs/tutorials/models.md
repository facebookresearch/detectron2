# Use Models

## Build Models from Yacs Config
From a yacs config object,
models (and their sub-models) can be built by
functions such as `build_model`, `build_backbone`, `build_roi_heads`:
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

`build_model` only builds the model structure and fills it with random parameters.
See below for how to load an existing checkpoint to the model and how to use the `model` object.

### Load/Save a Checkpoint
```python
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999")  # save to output/model_999.pth
```

Detectron2's checkpointer recognizes models in pytorch's `.pth` format, as well as the `.pkl` files
in our model zoo.
See [API doc](../modules/checkpoint.html#detectron2.checkpoint.DetectionCheckpointer)
for more details about its usage.

The model files can be arbitrarily manipulated using `torch.{load,save}` for `.pth` files or
`pickle.{dump,load}` for `.pkl` files.

### Use a Model

A model can be called by `outputs = model(inputs)`, where `inputs` is a `list[dict]`.
Each dict corresponds to one image and the required keys
depend on the type of model, and whether the model is in training or evaluation mode.
For example, in order to do inference,
all existing models expect the "image" key, and optionally "height" and "width".
The detailed format of inputs and outputs of existing models are explained below.

__Training__: When in training mode, all models are required to be used under an `EventStorage`.
The training statistics will be put into the storage:
```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
  losses = model(inputs)
```

__Inference__: If you only want to do simple inference using an existing model,
[DefaultPredictor](../modules/engine.html#detectron2.engine.defaults.DefaultPredictor)
is a wrapper around model that provides such basic functionality.
It includes default behavior including model loading, preprocessing,
and operates on single image rather than batches. See its documentation for usage.

You can also run inference directly like this:
```python
model.eval()
with torch.no_grad():
  outputs = model(inputs)
```

### Model Input Format

Users can implement custom models that support any arbitrary input format.
Here we describe the standard input format that all builtin models support in detectron2.
They all take a `list[dict]` as the inputs. Each dict
corresponds to information about one image.

The dict may contain the following keys:

* "image": `Tensor` in (C, H, W) format. The meaning of channels are defined by `cfg.INPUT.FORMAT`.
  Image normalization, if any, will be performed inside the model using
  `cfg.MODEL.PIXEL_{MEAN,STD}`.
* "height", "width": the **desired** output height and width **in inference**, which is not necessarily the same
  as the height or width of the `image` field.
  For example, the `image` field contains the resized image, if resize is used as a preprocessing step.
  But you may want the outputs to be in **original** resolution.
  If provided, the model will produce output in this resolution,
  rather than in the resolution of the `image` as input into the model. This is more efficient and accurate.
* "instances": an [Instances](../modules/structures.html#detectron2.structures.Instances)
  object for training, with the following fields:
  + "gt_boxes": a [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing N boxes, one for each instance.
  + "gt_classes": `Tensor` of long type, a vector of N labels, in range [0, num_categories).
  + "gt_masks": a [PolygonMasks](../modules/structures.html#detectron2.structures.PolygonMasks)
    or [BitMasks](../modules/structures.html#detectron2.structures.BitMasks) object storing N masks, one for each instance.
  + "gt_keypoints": a [Keypoints](../modules/structures.html#detectron2.structures.Keypoints)
    object storing N keypoint sets, one for each instance.
* "sem_seg": `Tensor[int]` in (H, W) format. The semantic segmentation ground truth for training.
  Values represent category labels starting from 0.
* "proposals": an [Instances](../modules/structures.html#detectron2.structures.Instances)
  object used only in Fast R-CNN style models, with the following fields:
  + "proposal_boxes": a [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing P proposal boxes.
  + "objectness_logits": `Tensor`, a vector of P scores, one for each proposal.

For inference of builtin models, only "image" key is required, and "width/height" are optional.

We currently don't define standard input format for panoptic segmentation training,
because models now use custom formats produced by custom data loaders.

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
  * "scores": `Tensor`, a vector of N confidence scores.
  * "pred_classes": `Tensor`, a vector of N labels in range [0, num_categories).
  + "pred_masks": a `Tensor` of shape (N, H, W), masks for each detected instance.
  + "pred_keypoints": a `Tensor` of shape (N, num_keypoint, 3).
    Each row in the last dimension is (x, y, score). Confidence scores are larger than 0.
* "sem_seg": `Tensor` of (num_categories, H, W), the semantic segmentation prediction.
* "proposals": [Instances](../modules/structures.html#detectron2.structures.Instances)
  object with the following fields:
  * "proposal_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes)
    object storing N boxes.
  * "objectness_logits": a torch vector of N confidence scores.
* "panoptic_seg": A tuple of `(pred: Tensor, segments_info: Optional[list[dict]])`.
  The `pred` tensor has shape (H, W), containing the segment id of each pixel.

  * If `segments_info` exists, each dict describes one segment id in `pred` and has the following fields:

    * "id": the segment id
    * "isthing": whether the segment is a thing or stuff
    * "category_id": the category id of this segment.

    If a pixel's id does not exist in `segments_info`, it is considered to be void label
    defined in [Panoptic Segmentation](https://arxiv.org/abs/1801.00868).

  * If `segments_info` is None, all pixel values in `pred` must be ≥ -1.
    Pixels with value -1 are assigned void labels.
    Otherwise, the category id of each pixel is obtained by
    `category_id = pixel // metadata.label_divisor`.


### Partially execute a model:

Sometimes you may want to obtain an intermediate tensor inside a model,
such as the input of certain layer, the output before post-processing.
Since there are typically hundreds of intermediate tensors, there isn't an API that provides you
the intermediate result you need.
You have the following options:

1. Write a (sub)model. Following the [tutorial](./write-models.md), you can
   rewrite a model component (e.g. a head of a model), such that it
   does the same thing as the existing component, but returns the output
   you need.
2. Partially execute a model. You can create the model as usual,
   but use custom code to execute it instead of its `forward()`. For example,
   the following code obtains mask features before mask head.

   ```python
   images = ImageList.from_tensors(...)  # preprocessed input tensor
   model = build_model(cfg)
   model.eval()
   features = model.backbone(images.tensor)
   proposals, _ = model.proposal_generator(images, features)
   instances, _ = model.roi_heads(images, features, proposals)
   mask_features = [features[f] for f in model.roi_heads.in_features]
   mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
   ```

3. Use [forward hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks).
   Forward hooks can help you obtain inputs or outputs of a certain module.
   If they are not exactly what you want, they can at least be used together with partial execution
   to obtain other tensors.

All options require you to read documentation and sometimes code
of the existing models to understand the internal logic,
in order to write code to obtain the internal tensors.
