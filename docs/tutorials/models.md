# Use Models

Models (and their sub-models) in detectron2 are built by
functions such as `build_model`, `build_backbone`, `build_roi_heads`:
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```


Next, we explain the input/output format used by the builtin models in detectron2.


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
