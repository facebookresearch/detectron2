import os


def resolve_images(coco_json, base_dir):
    images = coco_json["images"]
    for image in images:
        file_name = image["file_name"]
        image["file_name"] = os.path.join(base_dir, file_name)

    return coco_json