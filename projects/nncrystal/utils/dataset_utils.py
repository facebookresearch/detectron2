import os
from typing import List


def replace_image_base_dir(coco_object: dict, new_dir: str):
    images: List = coco_object["images"]
    for img in images:
        path: str = img["path"]
        new_path = os.path.join(new_dir, os.path.basename(path))
        img["path"] = new_path
    return coco_object
