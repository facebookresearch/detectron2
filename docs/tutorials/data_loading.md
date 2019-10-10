
# Using Custom Data Loaders

Detectron2 contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

Detectron2 provides two functions
`build_detection_{train,test}_loader` that create a data loader from a given config.
Here is how `build_detection_{train,test}_loader` work:

1. It takes the name of the dataset (e.g., "coco_2017_train") and loads a `list[dict]` representing the dataset items
   in a lightweight, canonical format. These dataset items are not yet ready to be used by the model (e.g., images are
   not loaded into memory, random augmentations have not been applied, etc.).
   Details about the dataset format and dataset registration can be found in [datasets](datasets).
2. Each dict in this list is mapped by a function ("mapper"):
	 * Users can customize this mapping function by specifying the "mapper" argument in
        `build_detection_{train,test}_loader`. The default mapper is [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper)
	 * There is no constraints on the output format, as long as it is accepted by the consumer of this data loader (usually the model).
   * The role of the mapper is to transform the lightweight, canonical representation of a dataset item into a format
     that is ready for the model to consume (including, e.g., reading images and performing random data augmentation).
	The output format of the default mapper is explained below.
3. The outputs of the mapper are batched (simply into a list).
4. This batched data is the output of the data loader. Typically, it's also the input of
	 `model.forward()`.


If you want to do something different (e.g., use different sampling or batching logic),
you can write your own data loader. The only requirement is that it produces the format your model accepts.
Next, we explain the input format used by the builtin models in detectron2.


### Model Input Format

The output of the default [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper) is a dict.
After the data loader performs batching, it becomes `list[dict]`, with one dict per image.
This will be the input format of all the builtin models.

The dict may contain the following keys:

* "image": `Tensor` in (C, H, W) format.
* "instances": an `Instances` object, with the following fields:
	+ "gt_boxes": `Boxes` object storing N boxes, one for each instance.
	+ "gt_classes": `Tensor`, a vector of N labels, in range [0, num_categories).
	+ "gt_masks": a `PolygonMasks` object storing N masks, one for each instance.
	+ "gt_keypoints": a `Keypoints` object storing N keypoint sets, one for each instance.
* "proposals": an `Instances` object used in Fast R-CNN style models, with the following fields:
	+ "proposal_boxes": `Boxes` object storing P proposal boxes.
	+ "objectness_logits": `Tensor`, a vector of P scores, one for each proposal.
* "height", "width": the *desired* output height and width of the image, not necessarily the same
	as the height or width of the `image` when input into the model, which might be after resizing.
	For example, it can be the *original* image height and width before resizing.

	If provided, the model will produce output in this resolution,
	rather than in the resolution of the `image` as input into the model. This is more efficient and accurate.
* "sem_seg": `Tensor[int]` in (H, W) format. The semantic segmentation ground truth.


### Model Output Format

The standard models outputs a `list[dict]`, one dict for each image. Each dict may contain:

* "instances": [Instances](../modules/structures.html#detectron2.structures.Instances)
  object with the following fields:
	* "pred_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes) object storing N boxes, one for each detected instance.
	* "scores": `Tensor`, a vector of N scores.
	* "pred_classes": `Tensor`, a vector of N labels in range [0, num_categories).
	+ "pred_masks": a `Tensor` of shape (N, H, W), masks for each detected instance.
	+ "pred_keypoints": a `Tensor` of shape (N, num_keypoint, 3).
		Each row in the last dimension is (x, y, score).
* "sem_seg": `Tensor` of (num_categories, H, W), the semantic segmentation prediction.
* "proposals": [Instances](../modules/structures.html#detectron2.structures.Instances)
	object with the following fields:
	* "proposal_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes)
		object storing N boxes.
	* "objectness_logits": a torch vector of N scores.
* "panoptic_seg": A tuple of (Tensor, list[dict]). The tensor has shape (H, W), where each element
	represent the segment id of the pixel. Each dict describes one segment id and has the following fields:
	* "id": the segment id
	* "isthing": whether the segment is a thing or stuff
	* "category_id": the category id of this segment. It represents the thing
       class id when `isthing==True`, and the stuff class id otherwise.
