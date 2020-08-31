
# Dataloader

Dataloader is the component that provides data to models.
A dataloader usually (but not necessarily) takes raw information from [datasets](./datasets.md),
and process them into a format needed by the model.

## How the Existing Dataloader Works

Detectron2 contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

Detectron2 provides two functions
[build_detection_{train,test}_loader](../modules/data.html#detectron2.data.build_detection_train_loader)
that create a default data loader from a given config.
Here is how `build_detection_{train,test}_loader` work:

1. It takes the name of a registered dataset (e.g., "coco_2017_train") and loads a `list[dict]` representing the dataset items
   in a lightweight format. These dataset items are not yet ready to be used by the model (e.g., images are
   not loaded into memory, random augmentations have not been applied, etc.).
   Details about the dataset format and dataset registration can be found in
   [datasets](./datasets.md).
2. Each dict in this list is mapped by a function ("mapper"):
   * Users can customize this mapping function by specifying the "mapper" argument in
        `build_detection_{train,test}_loader`. The default mapper is [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper).
   * The output format of the mapper can be arbitrary, as long as it is accepted by the consumer of this data loader (usually the model).
     The outputs of the default mapper, after batching, follow the default model input format documented in
     [Use Models](./models.html#model-input-format).
   * The role of the mapper is to transform the lightweight representation of a dataset item into a format
     that is ready for the model to consume (including, e.g., read images, perform random data augmentation and convert to torch Tensors).
     If you would like to perform custom transformations to data, you often want a custom mapper.
3. The outputs of the mapper are batched (simply into a list).
4. This batched data is the output of the data loader. Typically, it's also the input of
   `model.forward()`.


## Write a Custom Dataloader

Using a different "mapper" with `build_detection_{train,test}_loader(mapper=)` works for most use cases
of custom data loading.
For example, if you want to resize all images to a fixed size for training, use:

```python
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((800, 800))
   ]))
# use this dataloader instead of the default
```
If the arguments of the default [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper)
does not provide what you need, you may write a custom mapper function and use it instead, e.g.:

```python
from detectron2.data import detection_utils as utils
 # Show how to implement a minimal mapper, similar to the default DatasetMapper
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }
dataloader = build_detection_train_loader(cfg, mapper=mapper)
```

If you want to change not only the mapper (e.g., in order to implement different sampling or batching logic),
`build_detection_train_loader` won't work and you will need to write a different data loader.
The data loader is simply a
python iterator that produces [the format](./models.md) that the model accepts.
You can implement it using any tools you like.

No matter what to implement, it's recommended to
check out [API documentation of detectron2.data](../modules/data) to learn more about the APIs of
these functions.

## Use a Custom Dataloader

If you use [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer),
you can overwrite its `build_{train,test}_loader` method to use your own dataloader.
See the [deeplab dataloader](../../projects/DeepLab/train_net.py)
for an example.

If you write your own training loop, you can plug in your data loader easily.
