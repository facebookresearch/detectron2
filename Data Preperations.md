# Data Preparation for Detectron2

Detectron2 is open source object detection library by Facebook. We can train wide range of model's available in it. This tutorial is about training custorm model using Detectron2. 
In this way need dataset in coco formate. As we know, in object detection we need images along with ground truth area (i.e. bounding boxes around objects) and class of object.
Detectron2 needs COCO formate dataset specifically. COCO formate is descriped below 
```
{
  "info": {
    "year": "2021",
    "verison": "1",
    "description": "",
    "contributor": "Muhammad.Talha",
    "url": "",
    "date_created": "6/1/2021 12:00:00 AM"
  },
  "licenses": [
    {
      "id": 1,
      "url": "",
      "name": "Muhammad.Talha"
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "Dog",
      "supercategory": ""
    },
    {
      "id": 1,
      "name": "Cat",
      "supercategory": ""
    },
  ],
  "images": [
    {
      "id": 0,
      "license": 1,
      "file_name": "60fadcde6f158c9b44406d2b.jpeg",
      "height": 768,
      "width": 2732,
      "date_captured": "7/23/2021 9:00:00 AM"
    },
    ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [
        278,
        142,
        947,
        240
      ],
      "area": 227280,
      "segmentation": [
      ],
      "iscrowd": 0
    }
   ]
}
```
If you want to learn more about COCO annotation formate, you must read [Eric Hofesmann's](https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4#) blog. 

Here is question, how we can create this coco formate. There are multiple labling tools avaliable [Roboflow](http://roboflow.com/) and [LableImg](https://github.com/tzutalin/labelImg) etc.
