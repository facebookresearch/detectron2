
# Use Custom Dataloaders

## How the Existing Dataloader Works

Detectron2 contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

Detectron2 provides two functions
[build_detection_{train,test}_loader](../modules/data.html#detectron2.data.build_detection_train_loader)
that create a default data loader from a given config.
Here is how `build_detection_{train,test}_loader` work:

1. It takes the name of a registered dataset (e.g., "coco_2017_train") and loads a `list[dict]` representing the dataset items
   in a lightweight, canonical format. These dataset items are not yet ready to be used by the model (e.g., images are
   not loaded into memory, random augmentations have not been applied, etc.).
   Details about the dataset format and dataset registration can be found in
   [datasets](./datasets.md).
2. Each dict in this list is mapped by a function ("mapper"):
   * Users can customize this mapping function by specifying the "mapper" argument in
        `build_detection_{train,test}_loader`. The default mapper is [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper).
   * The output format of such function can be arbitrary, as long as it is accepted by the consumer of this data loader (usually the model).
     The outputs of the default mapper, after batching, follow the default model input format documented in
     [Use Models](./models.html#model-input-format).
   * The role of the mapper is to transform the lightweight, canonical representation of a dataset item into a format
     that is ready for the model to consume (including, e.g., read images, perform random data augmentation and convert to torch Tensors).
     If you would like to perform custom transformations to data, you often want a custom mapper.
3. The outputs of the mapper are batched (simply into a list).
4. This batched data is the output of the data loader. Typically, it's also the input of
   `model.forward()`.


## Write a Custom Dataloader

Using a different "mapper" with `build_detection_{train,test}_loader(mapper=)` works for most use cases
of custom data loading.
For example, if you want to resize all images to a fixed size for Mask R-CNN training, write this:

```python
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

def mapper(dataset_dict):
	# Implement a mapper, similar to the default DatasetMapper, but with your own customizations
	dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
	image = utils.read_image(dataset_dict["file_name"], format="BGR")
	image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
	dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

	annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
	instances = utils.annotations_to_instances(annos, image.shape[:2])
	dataset_dict["instances"] = utils.filter_empty_instances(instances)
	return dataset_dict

data_loader = build_detection_train_loader(cfg, mapper=mapper)
# use this dataloader instead of the default
```
Refer to [API documentation of detectron2.data](../modules/data) for details.

If you want to change not only the mapper (e.g., to write different sampling or batching logic),
you can write your own data loader. The data loader is simply a
python iterator that produces [the format](./models.md) your model accepts.
You can implement it using any tools you like.

## Use a Custom Dataloader

If you use [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer),
you can overwrite its `build_{train,test}_loader` method to use your own dataloader.
See the [densepose dataloader](../../projects/DensePose/train_net.py)
for an example.

If you write your own training loop, you can plug in your data loader easily.
