# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

from detectron2.data.catalog import Metadata
from detectron2.evaluation import COCOEvaluator

from densepose.data.datasets.coco import (
    get_contiguous_id_to_category_id_map,
    maybe_filter_categories_cocoapi,
)


def _maybe_add_iscrowd_annotations(cocoapi) -> None:
    for ann in cocoapi.dataset["annotations"]:
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0


class Detectron2COCOEvaluatorAdapter(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        output_dir=None,
        distributed=True,
    ):
        super().__init__(dataset_name, output_dir=output_dir, distributed=distributed)
        maybe_filter_categories_cocoapi(dataset_name, self._coco_api)
        _maybe_add_iscrowd_annotations(self._coco_api)
        # substitute category metadata to account for categories
        # that are mapped to the same contiguous id
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._maybe_substitute_metadata()

    def _maybe_substitute_metadata(self):
        cont_id_2_cat_id = get_contiguous_id_to_category_id_map(self._metadata)
        cat_id_2_cont_id = self._metadata.thing_dataset_id_to_contiguous_id
        if len(cont_id_2_cat_id) == len(cat_id_2_cont_id):
            return

        cat_id_2_cont_id_injective = {}
        for cat_id, cont_id in cat_id_2_cont_id.items():
            if (cont_id in cont_id_2_cat_id) and (cont_id_2_cat_id[cont_id] == cat_id):
                cat_id_2_cont_id_injective[cat_id] = cont_id

        metadata_new = Metadata(name=self._metadata.name)
        for key, value in self._metadata.__dict__.items():
            if key == "thing_dataset_id_to_contiguous_id":
                setattr(metadata_new, key, cat_id_2_cont_id_injective)
            else:
                setattr(metadata_new, key, value)
        self._metadata = metadata_new
