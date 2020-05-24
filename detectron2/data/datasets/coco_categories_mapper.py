# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Autogen with
# with open("lvis_v0.5_val.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
#     del x["image_count"]
#     del x["instance_count"]
# LVIS_CATEGORIES = repr(c) + "  # noqa"
from .coco_categories import *

def coco_cate_mapper():
#     print("Start mapping categories...")
#     print("Number of cat: ",len(LVIS_CATEGORIES))
    cate_type_map = {}
    for i in range(len(COCO_CATEGORIES)):
        # check the id is continuous
        assert COCO_CATEGORIES[i]['id'] == i+1
        if COCO_CATEGORIES[i]['frequency'] == 'f':
            cate_type_map[LVIS_CATEGORIES[i]['id']] = 0
        if COCO_CATEGORIES[i]['frequency'] == 'c':
            cate_type_map[LVIS_CATEGORIES[i]['id']] = 1
        if COCO_CATEGORIES[i]['frequency'] == 'r':
            cate_type_map[LVIS_CATEGORIES[i]['id']] = 2
    return cate_type_map

def cate_id_list():
    f_id_list = []
    c_id_list = []
    r_id_list = []
    fc_id_list = []
    fcr_id_list = []
    for i in range(len(LVIS_CATEGORIES)):
        assert LVIS_CATEGORIES[i]['id'] == i+1
        if LVIS_CATEGORIES[i]['frequency'] == 'f':
            f_id_list.append(LVIS_CATEGORIES[i]['id'])
            fc_id_list.append(LVIS_CATEGORIES[i]['id'])
            fcr_id_list.append(LVIS_CATEGORIES[i]['id'])
        if LVIS_CATEGORIES[i]['frequency'] == 'c':
            c_id_list.append(LVIS_CATEGORIES[i]['id'])
            fc_id_list.append(LVIS_CATEGORIES[i]['id'])
            fcr_id_list.append(LVIS_CATEGORIES[i]['id'])
        if LVIS_CATEGORIES[i]['frequency'] == 'r':
            r_id_list.append(LVIS_CATEGORIES[i]['id'])
            fcr_id_list.append(LVIS_CATEGORIES[i]['id'])
    return f_id_list, c_id_list, r_id_list, fc_id_list, fcr_id_list

if __name__ == "__main__":
    lvis_map = lvis_cate_mapper()
    print(lvis_map[1230])
