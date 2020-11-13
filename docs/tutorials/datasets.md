# Use Custom Datasets

This document explains how the dataset APIs
([DatasetCatalog](../modules/data.html#detectron2.data.DatasetCatalog), [MetadataCatalog](../modules/data.html#detectron2.data.MetadataCatalog))
work, and how to use them to add custom datasets.

Datasets that have builtin support in detectron2 are listed in [builtin datasets](builtin_datasets.md).
If you want to use a custom dataset while also reusing detectron2's data loaders,
you will need to:

1. __Register__ your dataset (i.e., tell detectron2 how to obtain your dataset).
2. Optionally, __register metadata__ for your dataset.

Next, we explain the above two concepts in detail.

The [Colab tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has a live example of how to register and train on a dataset of custom formats.

### Register a Dataset

To let detectron2 know how to obtain a dataset named "my_dataset", users need to implement
a function that returns the items in your dataset and then tell detectron2 about this
function:
```python
def my_dataset_function():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
# later, to access the data:
data: List[Dict] = DatasetCatalog.get("my_dataset")
```

Here, the snippet associates a dataset named "my_dataset" with a function that returns the data.
The function must return the same data if called multiple times.
The registration stays effective until the process exits.

The function can do arbitrary things and should return the data in either of the following formats:
1. Detectron2's standard dataset dict, described below. This will make it work with many other builtin
   features in detectron2, so it's recommended to use it when it's sufficient.
2. Any custom format. You can also return arbitrary dicts in your own format,
   such as adding extra keys for new tasks.
   Then you will need to handle them properly downstream as well.
   See below for more details.

#### Standard Dataset Dicts

For standard tasks
(instance detection, instance/semantic/panoptic segmentation, keypoint detection),
we load the original dataset into `list[dict]` with a specification similar to COCO's annotations.
This is our standard representation for a dataset.

Each dict contains information about one image.
The dict may have the following fields,
and the required fields vary based on what the dataloader or the task needs (see more below).

```eval_rst
.. list-table::
  :header-rows: 1

  * - Task
    - Fields
  * - Common
    - file_name, height, width, image_id

  * - Instance detection/segmentation
    - annotations

  * - Semantic segmentation
    - sem_seg_file_name

  * - Panoptic segmentation
    - pan_seg_file_name, segments_info
```

+ `file_name`: the full path to the image file. Rotation or flipping may be applied if the image has EXIF metadata.
+ `height`, `width`: integer. The shape of the image.
+ `image_id` (str or int): a unique id that identifies this image. Required by many
  evaluators to identify the images, but a dataset may use it for different purposes.
+ `annotations` (list[dict]): Required by __instance detection/segmentation or keypoint detection__ tasks.
  Each dict corresponds to annotations of one instance in this image, and
  may contain the following keys:
  + `bbox` (list[float], required): list of 4 numbers representing the bounding box of the instance.
  + `bbox_mode` (int, required): the format of bbox.  It must be a member of
    [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
    Currently supports: `BoxMode.XYXY_ABS`, `BoxMode.XYWH_ABS`.
  + `category_id` (int, required): an integer in the range [0, num_categories-1] representing the category label.
    The value num_categories is reserved to represent the "background" category, if applicable.
  + `segmentation` (list[list[float]] or dict): the segmentation mask of the instance.
    + If `list[list[float]]`, it represents a list of polygons, one for each connected component
      of the object. Each `list[float]` is one simple polygon in the format of `[x1, y1, ..., xn, yn]`.
      The Xs and Ys are absolute coordinates in unit of pixels.
    + If `dict`, it represents the per-pixel segmentation mask in COCO's compressed RLE format.
      The dict should have keys "size" and "counts". You can convert a uint8 segmentation mask of 0s and
      1s into such dict by `pycocotools.mask.encode(np.asarray(mask, order="F"))`.
      `cfg.INPUT.MASK_FORMAT` must be set to `bitmask` if using the default data loader with such format.
  + `keypoints` (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
    v[i] means the [visibility](http://cocodataset.org/#format-data) of this keypoint.
    `n` must be equal to the number of keypoint categories.
    The Xs and Ys are absolute real-value coordinates in range [0, W or H].

    (Note that the keypoint coordinates in COCO format are integers in range [0, W-1 or H-1], which is different
    from our standard format. Detectron2 adds 0.5 to COCO keypoint coordinates to convert them from discrete
    pixel indices to floating point coordinates.)
  + `iscrowd`: 0 (default) or 1. Whether this instance is labeled as COCO's "crowd
    region". Don't include this field if you don't know what it means.

  If `annotations` is an empty list, it means the image is labeled to have no objects.
  Such images will by default be removed from training,
  but can be included using `DATALOADER.FILTER_EMPTY_ANNOTATIONS`.

+ `sem_seg_file_name` (str):
  The full path to the semantic segmentation ground truth file.
  It should be a grayscale image whose pixel values are integer labels.
+ `pan_seg_file_name` (str):
  The full path to panoptic segmentation ground truth file.
  It should be an RGB image whose pixel values are integer ids encoded using the
  [panopticapi.utils.id2rgb](https://github.com/cocodataset/panopticapi/) function.
  The ids are defined by `segments_info`.
  If an id does not appear in `segments_info`, the pixel is considered unlabeled
  and is usually ignored in training & evaluation.
+ `segments_info` (list[dict]): defines the meaning of each id in panoptic segmentation ground truth.
  Each dict has the following keys:
  + `id` (int): integer that appears in the ground truth image.
  + `category_id` (int): an integer in the range [0, num_categories-1] representing the category label.
  + `iscrowd`: 0 (default) or 1. Whether this instance is labeled as COCO's "crowd region".


```eval_rst

.. note::

   The PanopticFPN model does not use the panoptic segmentation
   format defined here, but a combination of both instance segmentation and semantic segmentation data
   format. See :doc:`builtin_datasets` for instructions on COCO.

```

Fast R-CNN (with pre-computed proposals) models are rarely used today.
To train a Fast R-CNN, the following extra keys are needed:

+ `proposal_boxes` (array): 2D numpy array with shape (K, 4) representing K precomputed proposal boxes for this image.
+ `proposal_objectness_logits` (array): numpy array with shape (K, ), which corresponds to the objectness
 logits of proposals in 'proposal_boxes'.
+ `proposal_bbox_mode` (int): the format of the precomputed proposal bbox.
 It must be a member of
 [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
 Default is `BoxMode.XYXY_ABS`.



#### Custom Dataset Dicts for New Tasks

In the `list[dict]` that your dataset function returns, the dictionary can also have __arbitrary custom data__.
This will be useful for a new task that needs extra information not covered
by the standard dataset dicts. In this case, you need to make sure the downstream code can handle your data
correctly. Usually this requires writing a new `mapper` for the dataloader (see [Use Custom Dataloaders](./data_loading.md)).

When designing a custom format, note that all dicts are stored in memory
(sometimes serialized and with multiple copies).
To save memory, each dict is meant to contain __small__ but sufficient information
about each sample, such as file names and annotations.
Loading full samples typically happens in the data loader.

For attributes shared among the entire dataset, use `Metadata` (see below).
To avoid extra memory, do not save such information inside each sample.

### "Metadata" for Datasets

Each dataset is associated with some metadata, accessible through
`MetadataCatalog.get(dataset_name).some_metadata`.
Metadata is a key-value mapping that contains information that's shared among
the entire dataset, and usually is used to interpret what's in the dataset, e.g.,
names of classes, colors of classes, root of files, etc.
This information will be useful for augmentation, evaluation, visualization, logging, etc.
The structure of metadata depends on what is needed from the corresponding downstream code.

If you register a new dataset through `DatasetCatalog.register`,
you may also want to add its corresponding metadata through
`MetadataCatalog.get(dataset_name).some_key = some_value`, to enable any features that need the metadata.
You can do it like this (using the metadata key "thing_classes" as an example):

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

Here is a list of metadata keys that are used by builtin features in detectron2.
If you add your own dataset without these metadata, some features may be
unavailable to you:

* `thing_classes` (list[str]): Used by all instance detection/segmentation tasks.
  A list of names for each instance/thing category.
  If you load a COCO format dataset, it will be automatically set by the function `load_coco_json`.

* `thing_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each thing category.
  Used for visualization. If not given, random colors will be used.

* `stuff_classes` (list[str]): Used by semantic and panoptic segmentation tasks.
  A list of names for each stuff category.

* `stuff_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each stuff category.
  Used for visualization. If not given, random colors are used.

* `ignore_label` (int): Used by semantic and panoptic segmentation tasks. Pixels in ground-truth
  annotations with this category label should be ignored in evaluation. Typically these are "unlabeled"
  pixels.

* `keypoint_names` (list[str]): Used by keypoint detection. A list of names for each keypoint.

* `keypoint_flip_map` (list[tuple[str]]): Used by keypoint detection. A list of pairs of names,
  where each pair are the two keypoints that should be flipped if the image is
  flipped horizontally during augmentation.
* `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]. Each tuple specifies a pair of keypoints
  that are connected and the color to use for the line between them when visualized.

Some additional metadata that are specific to the evaluation of certain datasets (e.g. COCO):

* `thing_dataset_id_to_contiguous_id` (dict[int->int]): Used by all instance detection/segmentation tasks in the COCO format.
  A mapping from instance class ids in the dataset to contiguous ids in range [0, #class).
  Will be automatically set by the function `load_coco_json`.

* `stuff_dataset_id_to_contiguous_id` (dict[int->int]): Used when generating prediction json files for
  semantic/panoptic segmentation.
  A mapping from semantic segmentation class ids in the dataset
  to contiguous ids in [0, num_categories). It is useful for evaluation only.

* `json_file`: The COCO annotation json file. Used by COCO evaluation for COCO-format datasets.
* `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
* `evaluator_type`: Used by the builtin main training script to select
   evaluator. Don't use it in a new training script.
   You can just provide the [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)
   for your dataset directly in your main script.

```eval_rst
.. note::

   In recognition, sometimes we use the term "thing" for instance-level tasks,
   and "stuff" for semantic segmentation tasks.
   Both are used in panoptic segmentation tasks.
   For background on the concept of "thing" and "stuff", see
   `On Seeing Stuff: The Perception of Materials by Humans and Machines
   <http://persci.mit.edu/pub_pdfs/adelson_spie_01.pdf>`_.
```

### Register a COCO Format Dataset

If your instance-level (detection, segmentation, keypoint) dataset is already a json file in the COCO format,
the dataset and its associated metadata can be registered easily with:
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```

If your dataset is in COCO format but need to be further processed, or has extra custom per-instance annotations,
the [load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json)
function might be useful.

### Update the Config for New Datasets

Once you've registered the dataset, you can use the name of the dataset (e.g., "my_dataset" in
example above) in `cfg.DATASETS.{TRAIN,TEST}`.
There are other configs you might want to change to train or evaluate on new datasets:

* `MODEL.ROI_HEADS.NUM_CLASSES` and `MODEL.RETINANET.NUM_CLASSES` are the number of thing classes
  for R-CNN and RetinaNet models, respectively.
* `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` sets the number of keypoints for Keypoint R-CNN.
  You'll also need to set [Keypoint OKS](http://cocodataset.org/#keypoints-eval)
  with `TEST.KEYPOINT_OKS_SIGMAS` for evaluation.
* `MODEL.SEM_SEG_HEAD.NUM_CLASSES` sets the number of stuff classes for Semantic FPN & Panoptic FPN.
* `TEST.DETECTIONS_PER_IMAGE` controls the maximum number of objects to be detected.
  Set it to a larger number if test images may contain >100 objects.
* If you're training Fast R-CNN (with precomputed proposals), `DATASETS.PROPOSAL_FILES_{TRAIN,TEST}`
  need to match the datasets. The format of proposal files are documented
  [here](../modules/data.html#detectron2.data.load_proposals_into_dataset).

New models
(e.g. [TensorMask](../../projects/TensorMask),
[PointRend](../../projects/PointRend))
often have similar configs of their own that need to be changed as well.

```eval_rst
.. tip::

   After changing the number of classes, certain layers in a pre-trained model will become incompatible
   and therefore cannot be loaded to the new model.
   This is expected, and loading such pre-trained models will produce warnings about such layers.
```
