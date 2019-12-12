
# Use Custom Dataloaders

## How the Existing Dataloader Works

Detectron2 contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

Detectron2 provides two functions
[build_detection_{train,test}_loader](../modules/data.html#detectron2.data.build_detection_train_loader)
that create a default data loader from a given config.
Here is how `build_detection_{train,test}_loader` work:

1. It takes the name of the dataset (e.g., "coco_2017_train") and loads a `list[dict]` representing the dataset items
   in a lightweight, canonical format. These dataset items are not yet ready to be used by the model (e.g., images are
   not loaded into memory, random augmentations have not been applied, etc.).
   Details about the dataset format and dataset registration can be found in
   [datasets](datasets.html).
2. Each dict in this list is mapped by a function ("mapper"):
	 * Users can customize this mapping function by specifying the "mapper" argument in
        `build_detection_{train,test}_loader`. The default mapper is [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper)
	 * The output format of such function can be arbitrary, as long as it is accepted by the consumer of this data loader (usually the model).
   * The role of the mapper is to transform the lightweight, canonical representation of a dataset item into a format
     that is ready for the model to consume (including, e.g., read images, perform random data augmentation and convert to torch Tensors).
	 The output format of the default mapper is explained below.
3. The outputs of the mapper are batched (simply into a list).
4. This batched data is the output of the data loader. Typically, it's also the input of
   `model.forward()`.


## Write a Custom Dataloader

Using a different "mapper" with `build_detection_{train,test}_loader(mapper=)` works for most use cases
of custom data loading. Refer to [API documentation](../modules/data.html) for details.

If you want to do something different (e.g., use different sampling or batching logic),
you can write your own data loader. The data loader is simply a
python iterator that produces [the format](models.html) your model accepts. 
You can implement it using any tools you like.

## Use a Custom Dataloader

If you use [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer), 
you can overwrite its `build_{train,test}__loader` method to use your own dataloader.
See the [densepose dataloader](/projects/DensePose/train_net.py)
for an example.

If you write your own training loop, you can plug in your data loader easily.
