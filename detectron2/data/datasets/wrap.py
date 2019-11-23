from .. import DatasetCatalog, MetadataCatalog

def load_wrapped_dataset(source_dataset_name, target_dataset_name):
    # load the source dataset dicts
    dataset_dicts = DatasetCatalog.get(source_dataset_name)

    # load category id mapping
    wrap_name = f"{source_dataset_name}@{target_dataset_name}"
    idS_to_idT = MetadataCatalog.get(wrap_name).idS_to_idT

    wrap_dataset_dicts(dataset_dicts, idS_to_idT)
    return dataset_dicts


def get_id_mapping(source_class_names, target_class_names):
    common_classes = set(source_class_names) & set(target_class_names)
    assert len(common_classes) > 0

    idS_to_idT = {
        source_class_names.index(name): target_class_names.index(name)
        for name in common_classes
    }

    return idS_to_idT


def wrap_dataset_dicts(dataset_dicts, idS_to_idT, drop_empty=True):
    for d in dataset_dicts:
        objects = d["annotations"]
        wrapped_objects = []
        for obj in objects:
            # if drop_empty is False, categories that has no corresponding
            # mappings will have None values as category ids and have to be
            # handled by the User.
            wrapped_category_id = idS_to_idT.get(obj["category_id"], None)
            if wrapped_category_id is not None or not drop_empty:
                wrapped_objects.append(wrapped_category_id)

        d["annotations"] = wrapped_objects


def register_wrapper(source_dataset_name, target_dataset_name, default="unlabeled"):
    wrap_name = f"{source_dataset_name}@{target_dataset_name}"
    DatasetCatalog.register(
        wrap_name,
        lambda: load_wrapped_dataset(source_dataset_name, target_dataset_name)
    )

    # Copy all metadata from source
    source_metadata_dict = MetadataCatalog.get(source_dataset_name).as_dict()
    source_metadata_dict.pop("name", None)
    source_metadata_dict.pop("thing_classes", None)
    MetadataCatalog.get(wrap_name).set(**source_metadata_dict)

    # Load metadata for mapping categories
    source_metadata = MetadataCatalog.get(source_dataset_name)
    target_metadata = MetadataCatalog.get(target_dataset_name)

    # Add source -> target category_id mapping to metadata
    source_class_names = source_metadata.thing_classes
    target_class_names = target_metadata.thing_classes
    idS_to_idT = get_id_mapping(source_class_names, target_class_names)
    MetadataCatalog.get(wrap_name).idS_to_idT = idS_to_idT

    # Add target thing_classes to metadata
    wrapped_class_names = []
    for category_id, class_name in enumerate(target_class_names):
        if category_id not in idS_to_idT.values():
            # Different ways to handle source classes that are not
            # present in the target class set

            # 1) Decorate unmatched target classes
            # class_name += " (unmatched)"

            # 2) Use a default value
            class_name = default

        wrapped_class_names.append(class_name)
    MetadataCatalog.get(wrap_name).thing_classes = wrapped_class_names
