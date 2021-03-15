# Copyright (c) Facebook, Inc. and its affiliates.
from .chimpnsee import register_dataset as register_chimpnsee_dataset
from .coco import BASE_DATASETS as BASE_COCO_DATASETS
from .coco import DATASETS as COCO_DATASETS
from .coco import register_datasets as register_coco_datasets
from .lvis import DATASETS as LVIS_DATASETS
from .lvis import register_datasets as register_lvis_datasets

DEFAULT_DATASETS_ROOT = "datasets"


# pyre-fixme[6]: Expected `Optional[_PathLike[typing.Any]]` for 2nd param but got `str`.
# pyre-fixme[6]: Expected `Optional[_PathLike[typing.Any]]` for 2nd param but got `str`.
# pyre-fixme[6]: Expected `Optional[_PathLike[typing.Any]]` for 2nd param but got `str`.
register_coco_datasets(COCO_DATASETS, DEFAULT_DATASETS_ROOT)
# pyre-fixme[6]: Expected `Optional[_PathLike[typing.Any]]` for 2nd param but got `str`.
register_coco_datasets(BASE_COCO_DATASETS, DEFAULT_DATASETS_ROOT)
# pyre-fixme[6]: Expected `Optional[_PathLike[typing.Any]]` for 2nd param but got `str`.
register_lvis_datasets(LVIS_DATASETS, DEFAULT_DATASETS_ROOT)

# pyre-fixme[19]: Expected 0 positional arguments.
register_chimpnsee_dataset(DEFAULT_DATASETS_ROOT)
