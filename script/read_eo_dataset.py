import os
import json
import glob
import numpy as np
from PIL import Image

from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.annotation.annotion_converters import RotatedBBoxConverter

DATASET_PATH_ROOT = "/home/ubuntu/Documents/dataset/Aerial_dataset/EO-Detection"
IMAGE_DIR = "images"
LABEL_DIR = "label"


def read_eo_dataset():
    global DATASET_PATH_ROOT
    global IMAGE_DIR
    global LABEL_DIR

    root_path = DATASET_PATH_ROOT
    image_dir = IMAGE_DIR
    label_dir = LABEL_DIR

    rotated_bbox_cvt = RotatedBBoxConverter()

    label_list = glob.glob(os.path.join(root_path, label_dir, "*.json"))

    with open(label_list[0], 'r') as json_file:
        json_labels = json.load(json_file)

    labels = json_labels["features"]

    annotations = []
    for label in labels:
        filepath = os.path.join(root_path, image_dir, label["properties"]["image_id"])
        image = Image.open(filepath)

        image_width, image_height = image.size
        classes = label["properties"]["type_name"]

        rbox = label["properties"]["bounds_imcoords"].split(",")
        p1 = (float(rbox[0]), float(rbox[1]))
        p2 = (float(rbox[2]), float(rbox[3]))
        p3 = (float(rbox[4]), float(rbox[5]))
        p4 = (float(rbox[6]), float(rbox[7]))

        # processing Label Noise
        if p1[0] > p4[0]:
            p1, p2, p3, p4 = p3, p4, p1, p2

        homo_coord_points = np.array([np.append(p1, [1]),
                                      np.append(p2, [1]),
                                      np.append(p3, [1]),
                                      np.append(p4, [1])])

        bbox, theta = rotated_bbox_cvt.get_rotated_bbox(homo_coord_points)

        annotations.append({
            "filepath": filepath,
            "image_width": image_width,
            "image_height": image_height,
            "objects": [{"class": classes,
                         "xmin": bbox[0],
                         "ymin": bbox[1],
                         "xmax": bbox[2],
                         "ymax": bbox[3],
                         "theta": -theta}]
        })

    return annotations


if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)
    detection_anno.dump()
