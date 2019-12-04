import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
import random

# write a function that loads the dataset into detectron2's standard format
def get_icdar_dicts(img_dir,train):
    json_file = os.path.join(img_dir, "train_labels.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
#     for _, v in imgs_anns.items():
    for each_key in list(imgs_anns.keys())[:50]:
        v=imgs_anns[each_key]
        record = {}
        
        filename = os.path.join(img_dir,'train_images', '%s.jpg'%each_key)
#         print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
#         print(v)
#         annos = v["regions"]
        objs = []
        for  anno in v:
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
            px = np.array(anno['points'])[:,0]
            py = np.array(anno['points'])[:,1]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    random.seed(1)
    random.shuffle(dataset_dicts)
    train_num=int(len(dataset_dicts)*0.8)
#     print(train_num,len(dataset_dicts))
    if train:
        return dataset_dicts[:train_num]
    else:
    
        return dataset_dicts[train_num:]

from detectron2.data import DatasetCatalog, MetadataCatalog
# for d in ["train", "val"]:
# dataset_dicts=get_icdar_dicts('/input0')
d='train'
DatasetCatalog.register("icdar_" + d, lambda d=d: get_icdar_dicts('/input0',True))
MetadataCatalog.get("icdar_" + d).set(thing_classes=["text"])
d='val'
DatasetCatalog.register("icdar_" + d, lambda d=d:get_icdar_dicts('/input0',False) )
MetadataCatalog.get("icdar_" + d).set(thing_classes=["text"])

balloon_metadata = MetadataCatalog.get("/input0/train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ['icdar_train']
cfg.DATASETS.TEST = ['icdar_val']   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()