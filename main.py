from custom import Trainer
import math
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from collections import Counter
from detectron2.structures import BoxMode
from script.read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from typing import List

def collect_class_info(anno):
    objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
    classes_info = Counter([obj.CLASS for obj in objs])
    classes_list = list(classes_info.keys())

    return classes_list


def bbox_convert(
    xmin: float, ymin: float, xmax: float, ymax: float, theta: float
) -> List:

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    center_x = xmin + (bbox_width / 2)
    center_y = ymin + (bbox_height / 2)

    return [center_x, center_y, bbox_width, bbox_height, math.degrees(theta)]


def get_eo_detection_dicts():
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)
    classes_list = collect_class_info(detection_anno)

    dataset_dicts = []
    for idx, FILE in enumerate(detection_anno.FILES):
        record = {}

        filename = FILE.FILEPATH
        height = FILE.IMAGE_HEIGHT
        width = FILE.IMAGE_WIDTH

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        for OBJ in FILE.OBJECTS:
            xmin = OBJ.XMIN
            ymin = OBJ.YMIN
            xmax = OBJ.XMAX
            ymax = OBJ.YMAX
            theta = -OBJ.THETA

            classes = OBJ.CLASS

            cx, cy, bbox_w, bbox_h, angle = bbox_convert(xmin, ymin, xmax, ymax, theta)
            objs.append(
                {
                    "bbox": [cx, cy, bbox_w, bbox_h, angle],
                    "bbox_mode": BoxMode.XYWHA_ABS,
                    "category_id": classes_list.index(classes),
                    "iscrowd": 0,
                }
            )

            record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts


cfg = get_cfg()

# Model Archtiecture
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
)
cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0, 45, 90, 135, 180]]
cfg.MODEL.MASK_ON = False
cfg.MODEL.KEYPOINT_ON = False
cfg.DATASETS.TRAIN = ("EO-Detection",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.BASE_LR = 0.00025


DatasetCatalog.register("EO-Detection", get_eo_detection_dicts)
MetadataCatalog.get("EO-Detection").set(
    thing_classes=["maritime vessels", "container", "oil tanker", "aircraft carrier"]
)
eo_detection_metadata = MetadataCatalog.get("EO-Detection")

dataset_dicts = get_eo_detection_dicts()


if __name__ == "__main__":

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
