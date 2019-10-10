
For a few datasets that detectron2 natively supports,
the datasets are assumed to exist in a directory called
"datasets/", under the directory where you launch the program.
with the following directory structure:

## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./prepare_for_tests.sh`.

## Expected dataset structure for PanopticFPN:

```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/
    # png annotations
```

Install panopticapi by:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```
Then, run `./prepare_panoptic_fpn.py`, to extract semantic annotations from panoptic annotations.

## Expected dataset structure for LVIS instance detection/segmentation:
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
	lvis_v0.5_image_info_test.json
```

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json
				labelTrainIds.png (created by cityscapesscripts/preparation/createTrainIdLabelImgs.py)
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
	ImageSets/
	JPEGImages/
```
