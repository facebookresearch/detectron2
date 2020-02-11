# Use Custom Datasets

If you want to use a custom dataset while also reusing detectron2's data loaders,
you will need to

1. Register your dataset (i.e., tell detectron2 how to obtain your dataset).
2. Optionally, register metadata for your dataset.

Next, we explain the above two concepts in details.

The [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has a working example of how to register and train on a dataset of custom formats.


### Register a Dataset

To let detectron2 know how to obtain a dataset named "my_dataset", you will implement
a function that returns the items in your dataset and then tell detectron2 about this
function:
```python
def get_dicts():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", get_dicts)
```

Here, the snippet associates a dataset "my_dataset" with a function that returns the data.
If you do not modify downstream code (i.e., you use the standard data loader and data mapper),
then the function has to return a list of dicts in detectron2's standard dataset format, described
next. You can also use arbitrary custom data format, as long as the
downstream code (mainly the [custom data loader](data_loading.html)) supports it.

For standard tasks
(instance detection, instance/semantic/panoptic segmentation, keypoint detection),
we use a format similar to COCO's json annotations
as the basic dataset representation.

The format uses one dict to represent the annotations of
one image. The dict may have the following fields.
The fields are often optional, and some functions may be able to
infer certain fields from others if needed, e.g., the data loader
will load the image from "file_name" and load "sem_seg" from "sem_seg_file_name".

+ `file_name`: the full path to the image file. Will apply rotation and flipping if the image has such exif information.
+ `sem_seg_file_name`: the full path to the ground truth semantic segmentation file.
+ `sem_seg`: semantic segmentation ground truth in a 2D `torch.Tensor`. Values in the array represent
   category labels starting from 0.
+ `height`, `width`: integer. The shape of image.
+ `image_id` (str or int): a unique id that identifies this image. Used
	during evaluation to identify the images, but a dataset may use it for different purposes.
+ `annotations` (list[dict]): each dict corresponds to annotations of one instance
  in this image. Images with empty `annotations` will by default be removed from training,
	but can be included using `DATALOADER.FILTER_EMPTY_ANNOTATIONS`.
	Each dict may contain the following keys:
  + `bbox` (list[float]): list of 4 numbers representing the bounding box of the instance.
  + `bbox_mode` (int): the format of bbox.
    It must be a member of
    [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
    Currently supports: `BoxMode.XYXY_ABS`, `BoxMode.XYWH_ABS`.
  + `category_id` (int): an integer in the range [0, num_categories) representing the category label.
    The value num_categories is reserved to represent the "background" category, if applicable.
  + `segmentation` (list[list[float]] or dict):
    + If `list[list[float]]`, it represents a list of polygons, one for each connected component
      of the object. Each `list[float]` is one simple polygon in the format of `[x1, y1, ..., xn, yn]`.
      The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
      depend on whether "bbox_mode" is relative.
    + If `dict`, it represents the per-pixel segmentation mask in COCO's RLE format. The dict should have
			keys "size" and "counts". You can convert a uint8 segmentation mask of 0s and 1s into
			RLE format by `pycocotools.mask.encode(np.asarray(mask, order="F"))`.
  + `keypoints` (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
    v[i] means the [visibility](http://cocodataset.org/#format-data) of this keypoint.
    `n` must be equal to the number of keypoint categories.
    The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
    depend on whether "bbox_mode" is relative.

    Note that the coordinate annotations in COCO format are integers in range [0, H-1 or W-1].
    By default, detectron2 adds 0.5 to absolute keypoint coordinates to convert them from discrete
    pixel indices to floating point coordinates.
  + `iscrowd`: 0 or 1. Whether this instance is labeled as COCO's "crowd
    region". Don't include this field if you don't know what it means.

The following keys are used by Fast R-CNN style training, which is rare today.

+ `proposal_boxes` (array): 2D numpy array with shape (K, 4) representing K precomputed proposal boxes for this image.
+ `proposal_objectness_logits` (array): numpy array with shape (K, ), which corresponds to the objectness
  logits of proposals in 'proposal_boxes'.
+ `proposal_bbox_mode` (int): the format of the precomputed proposal bbox.
  It must be a member of
  [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
  Default format is `BoxMode.XYXY_ABS`.


If your dataset is already in the COCO format, you can simply register it by
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```
which will take care of everything (including metadata) for you.

If your dataset is in COCO format with custom per-instance annotations,
the [load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json) function can be used.


### "Metadata" for Datasets

Each dataset is associated with some metadata, accessible through
`MetadataCatalog.get(dataset_name).some_metadata`.
Metadata is a key-value mapping that contains primitive information that helps interpret what's in the dataset, e.g.,
names of classes, colors of classes, root of files, etc.
This information will be useful for augmentation, evaluation, visualization, logging, etc.
The structure of metadata depends on the what is needed from the corresponding downstream code.


If you register a new dataset through `DatasetCatalog.register`,
you may also want to add its corresponding metadata through
`MetadataCatalog.get(dataset_name).set(name, value)`, to enable any features that need metadata.
You can do it like this (using the metadata field "thing_classes" as an example):

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
  Used for visualization. If not given, random colors are used.

* `stuff_classes` (list[str]): Used by semantic and panoptic segmentation tasks.
  A list of names for each stuff category.

* `stuff_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each stuff category.
  Used for visualization. If not given, random colors are used.

* `keypoint_names` (list[str]): Used by keypoint localization. A list of names for each keypoint.

* `keypoint_flip_map` (list[tuple[str]]): Used by the keypoint localization task. A list of pairs of names,
  where each pair are the two keypoints that should be flipped if the image is
  flipped during augmentation.
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
   evaluator. No need to use it if you write your own main script.
   You can just provide the [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)
   for your dataset directly in your main script.

NOTE: For background on the concept of "thing" and "stuff", see
[On Seeing Stuff: The Perception of Materials by Humans and Machines](http://persci.mit.edu/pub_pdfs/adelson_spie_01.pdf).
In detectron2, the term "thing" is used for instance-level tasks,
and "stuff" is used for semantic segmentation tasks.
Both are used in panoptic segmentation.


### Update the Config for New Datasets

Once you've registered the dataset, you can use the name of the dataset (e.g., "my_dataset" in
example above) in `DATASETS.{TRAIN,TEST}`.
There are other configs you might want to change to train or evaluate on new datasets:

* `MODEL.ROI_HEADS.NUM_CLASSES` and `MODEL.RETINANET.NUM_CLASSES` are the number of thing classes
	for R-CNN and RetinaNet models.
* `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` sets the number of keypoints for Keypoint R-CNN.
  You'll also need to set [Keypoint OKS](http://cocodataset.org/#keypoints-eval)
	with `TEST.KEYPOINT_OKS_SIGMAS` for evaluation.
* `MODEL.SEM_SEG_HEAD.NUM_CLASSES` sets the number of stuff classes for Semantic FPN & Panoptic FPN.
* If you're training Fast R-CNN (with precomputed proposals), `DATASETS.PROPOSAL_FILES_{TRAIN,TEST}`
	need to match the datasts. The format of proposal files are documented
	[here](../modules/data.html#detectron2.data.load_proposals_into_dataset).
