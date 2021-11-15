import os, random, cv2
import supervisely_lib as sly
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


my_app = sly.AppService()

api: sly.Api = my_app.public_api
task_id = my_app.task_id

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

work_dir = os.path.join(my_app.cache_dir, "detectron")

project_info = api.project.get_info_by_id(project_id)
meta_json = api.project.get_meta(project_info.id)
meta = sly.ProjectMeta.from_json(meta_json)

all_classes = {}

for class_index, obj_class in enumerate(meta.obj_classes):
    all_classes[obj_class.name] = class_index


def get_sl_dicts(project_id):

    project_info = api.project.get_info_by_id(project_id)
    meta_json = api.project.get_meta(project_info.id)
    meta = sly.ProjectMeta.from_json(meta_json)

    dataset_dicts = []
    datasets = api.dataset.get_list(project_id)

    for dataset in datasets:
        images = api.image.get_list(dataset.id, sort="name")
        for batch in sly.batched(images, batch_size=3):

            image_ids = [image_info.id for image_info in batch]
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]

            for idx, img_info in enumerate(batch):
                api.image.download_path(img_info.id, os.path.join(work_dir, img_info.name))
                record = {}
                record["file_name"] = os.path.join(work_dir, img_info.name)
                record["image_id"] = img_info.id
                record["height"] = img_info.height
                record["width"] = img_info.width

                ann = sly.Annotation.from_json(ann_jsons[idx], meta)
                objs = []
                for label in ann.labels:

                    rect = label.geometry.to_bbox()

                    obj = {
                        "bbox": [rect.left, rect.top, rect.right, rect.bottom],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": all_classes[label.obj_class.name],
                    }

                    objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)

    return dataset_dicts


DatasetCatalog.register("lemons", get_sl_dicts)
MetadataCatalog.get("lemons").thing_classes = list(all_classes.keys())

balloon_metadata = MetadataCatalog.get("lemons")

dataset_dicts = get_sl_dicts(project_id)

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('', out.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        break

