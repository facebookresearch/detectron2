import argparse
import json
import logging
from typing import List, Dict


def merge_coco_datasets(coco_dicts: List[Dict], mapping_hint: Dict):
    """
    Merge multiple coco dicts
    :param coco_dicts:
    :param mapping_hint: dictionary containing {"category": id} pairs.
    If category present in coco_dicts does not exist in mapping_hint, raise ValueError.
    If None, the mapping will be built by the order of category occurrence.
    :return: merged dataset. The first dataset's info entry is used for the output
    """
    assert len(coco_dicts)
    ret_dict = {
        "info": coco_dicts[0]["info"],
        "categories": [],
        "license": [],
        "images": [],
        "annotations": [],
    }

    # merge category
    if mapping_hint is None:
        category_id_offset = 1 # category id start from 1
        mapping_hint = {}
        for coco_dict in coco_dicts:
            cats = coco_dict["categories"]
            for cat in cats:
                if cat["name"] not in coco_dict:
                    # The original id is discarded. Rebuild with the occurrence order.
                    cat_copy = cat.copy()
                    cat_copy["id"] = category_id_offset
                    mapping_hint[cat["name"]] = category_id_offset
                    category_id_offset += 1

                    ret_dict["categories"].append(cat_copy)

    image_id_offset = 0
    annotation_id_offset = 1 # start from 1
    for coco_dict in coco_dicts:
        # from first dataset, merge images with concatenated ids
        new_images = coco_dict["images"]
        image_id_mapping = {}
        for image in new_images:
            # Note, image id may not be contiguous? Discard the original id and rebuild with occurrence order.
            image_id_mapping[image["id"]] = image_id_offset
            image["id"] = image_id_offset
            image_id_offset += 1

        # cat mapping from original to merged
        cat_mapping = {}
        cats = coco_dict["categories"]
        for cat in cats:
            old_id = cat["id"]
            old_name = cat["name"]
            new_id = mapping_hint[old_name]
            cat_mapping[old_id] = new_id

        # annotations' image_id should be updated accordingly
        new_anns = coco_dict["annotations"]
        for ann in new_anns:
            ann["id"] = annotation_id_offset
            annotation_id_offset += 1
            ann["image_id"] = image_id_mapping[ann["image_id"]]
            ann["category_id"] = cat_mapping[ann["category_id"]]

        # merge new datasets
        ret_dict["images"].extend(new_images)
        if "license" in coco_dict:
            ret_dict["license"].extend(coco_dict["license"])
        ret_dict["annotations"].extend(new_anns)

    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=str, required=True)
    parser.add_argument("--mapping_hint")
    parser.add_argument("--output", type=str, default="merged_coco_dataset.json")
    args = parser.parse_args()

    assert len(args.input)
    datasets = []
    for i, input_file in enumerate(args.input):
        with open(input_file, "r") as f:
            logging.info(f"Loading dataset {i}: {input_file}")
            datasets.append(json.load(f))

    mapping_hint = json.loads(args.mapping_hint) if args.mapping_hint is not None else None

    new_dataset = merge_coco_datasets(datasets, mapping_hint)

    logging.info(f"Writing output file {args.output}")
    with open(args.output, "w") as f:
        json.dump(new_dataset, f)
