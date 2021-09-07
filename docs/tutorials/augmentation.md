
# Data Augmentation

Augmentation is an important part of training.
Detectron2's data augmentation system aims at addressing the following goals:

1. Allow augmenting multiple data types together
   (e.g., images together with their bounding boxes and masks)
2. Allow applying a sequence of statically-declared augmentation
3. Allow adding custom new data types to augment (rotated bounding boxes, video clips, etc.)
4. Process and manipulate the __operations__ that are applied by augmentations

The first two features cover most of the common use cases, and is also
available in other libraries such as [albumentations](https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e).
Supporting other features adds some overhead to detectron2's augmentation API,
which we'll explain in this tutorial.

This tutorial focuses on how to use augmentations when writing new data loaders,
and how to write new augmentations.
If you use the default data loader in detectron2, it already supports taking a user-provided list of custom augmentations,
as explained in the [Dataloader tutorial](data_loading).

## Basic Usage

The basic usage of feature (1) and (2) is like the following:
```python
from detectron2.data import transforms as T
# Define a sequence of augmentations:
augs = T.AugmentationList([
    T.RandomBrightness(0.9, 1.1),
    T.RandomFlip(prob=0.5),
    T.RandomCrop("absolute", (640, 640))
])  # type: T.Augmentation

# Define the augmentation input ("image" required, others optional):
input = T.AugInput(image, boxes=boxes, sem_seg=sem_seg)
# Apply the augmentation:
transform = augs(input)  # type: T.Transform
image_transformed = input.image  # new image
sem_seg_transformed = input.sem_seg  # new semantic segmentation

# For any extra data that needs to be augmented together, use transform, e.g.:
image2_transformed = transform.apply_image(image2)
polygons_transformed = transform.apply_polygons(polygons)
```

Three basic concepts are involved here. They are:
* [T.Augmentation](../modules/data_transforms.html#detectron2.data.transforms.Augmentation) defines the __"policy"__ to modify inputs.
  * its `__call__(AugInput) -> Transform` method augments the inputs in-place, and returns the operation that is applied
* [T.Transform](../modules/data_transforms.html#detectron2.data.transforms.Transform)
  implements the actual __operations__ to transform data
  * it has methods such as `apply_image`, `apply_coords` that define how to transform each data type
* [T.AugInput](../modules/data_transforms.html#detectron2.data.transforms.AugInput)
  stores inputs needed by `T.Augmentation` and how they should be transformed.
  This concept is needed for some advanced usage.
  Using this class directly should be sufficient for all common use cases,
  since extra data not in `T.AugInput` can be augmented using the returned
  `transform`, as shown in the above example.

## Write New Augmentations

Most 2D augmentations only need to know about the input image. Such augmentation can be implemented easily like this:

```python
class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)

class MyCustomResize(T.Augmentation):
    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        new_h, new_w = int(old_h * np.random.rand()), int(old_w * 1.5)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)

augs = MyCustomResize()
transform = augs(input)
```

In addition to image, any attributes of the given `AugInput` can be used as long
as they are part of the function signature, e.g.:

```python
class MyCustomCrop(T.Augmentation):
    def get_transform(self, image, sem_seg):
        # decide where to crop using both image and sem_seg
        return T.CropTransform(...)

augs = MyCustomCrop()
assert hasattr(input, "image") and hasattr(input, "sem_seg")
transform = augs(input)
```

New transform operation can also be added by subclassing
[T.Transform](../modules/data_transforms.html#detectron2.data.transforms.Transform).

## Advanced Usage

We give a few examples of advanced usages that
are enabled by our system.
These options can be interesting to new research,
although changing them is often not needed
for standard use cases.

### Custom transform strategy

Instead of only returning the augmented data, detectron2's `Augmentation` returns the __operations__ as `T.Transform`.
This allows users to apply custom transform strategy on their data.
We use keypoints data as an example.

Keypoints are (x, y) coordinates, but they are not so trivial to augment due to the semantic meaning they carry.
Such meaning is only known to the users, therefore users may want to augment them manually
by looking at the returned `transform`.
For example, when an image is horizontally flipped, we'd like to swap the keypoint annotations for "left eye" and "right eye".
This can be done like this (included by default in detectron2's default data loader):
```python
# augs, input are defined as in previous examples
transform = augs(input)  # type: T.Transform
keypoints_xy = transform.apply_coords(keypoints_xy)   # transform the coordinates

# get a list of all transforms that were applied
transforms = T.TransformList([transform]).transforms
# check if it is flipped for odd number of times
do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms) % 2 == 1
if do_hflip:
    keypoints_xy = keypoints_xy[flip_indices_mapping]
```

As another example, keypoints annotations often have a "visibility" field.
A sequence of augmentations might augment a visible keypoint out of the image boundary (e.g. with cropping),
but then bring it back within the boundary afterwards (e.g. with image padding).
If users decide to label such keypoints "invisible",
then the visibility check has to happen after every transform step.
This can be achieved by:

```python
transform = augs(input)  # type: T.TransformList
assert isinstance(transform, T.TransformList)
for t in transform.transforms:
    keypoints_xy = t.apply_coords(keypoints_xy)
    visibility &= (keypoints_xy >= [0, 0] & keypoints_xy <= [W, H]).all(axis=1)

# btw, detectron2's `transform_keypoint_annotations` function chooses to label such keypoints "visible":
# keypoints_xy = transform.apply_coords(keypoints_xy)
# visibility &= (keypoints_xy >= [0, 0] & keypoints_xy <= [W, H]).all(axis=1)
```


### Geometrically invert the transform
If images are pre-processed by augmentations before inference, the predicted results
such as segmentation masks are localized on the augmented image.
We'd like to invert the applied augmentation with the [inverse()](../modules/data_transforms.html#detectron2.data.transforms.Transform.inverse)
API, to obtain results on the original image:
```python
transform = augs(input)
pred_mask = make_prediction(input.image)
inv_transform = transform.inverse()
pred_mask_orig = inv_transform.apply_segmentation(pred_mask)
```

### Add new data types

[T.Transform](../modules/data_transforms.html#detectron2.data.transforms.Transform)
supports a few common data types to transform, including images, coordinates, masks, boxes, polygons.
It allows registering new data types, e.g.:
```python
@T.HFlipTransform.register_type("rotated_boxes")
def func(flip_transform: T.HFlipTransform, rotated_boxes: Any):
    # do the work
    return flipped_rotated_boxes

t = HFlipTransform(width=800)
transformed_rotated_boxes = t.apply_rotated_boxes(rotated_boxes)  # func will be called
```

### Extend T.AugInput

An augmentation can only access attributes available in the given input.
[T.AugInput](../modules/data_transforms.html#detectron2.data.transforms.StandardAugInput) defines "image", "boxes", "sem_seg",
which are sufficient for common augmentation strategies to decide how to augment.
If not, a custom implementation is needed.

By re-implement the "transform()" method in AugInput, it is also possible to
augment different fields in ways that are dependent on each other.
Such use case is uncommon (e.g. post-process bounding box based on augmented masks), but allowed by the system.

