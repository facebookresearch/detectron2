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


def get_sl_dicts():

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
                    curr_poly = label.geometry.exterior_np.tolist()
                    new_poly = [point[::-1] for point in curr_poly ]

                    obj = {
                        "bbox": [rect.left, rect.top, rect.right, rect.bottom],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [new_poly],
                        "category_id": all_classes[label.obj_class.name],
                    }

                    objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)

    return dataset_dicts


DatasetCatalog.register("lemons", get_sl_dicts)
MetadataCatalog.get("lemons").thing_classes = list(all_classes.keys())

balloon_metadata = MetadataCatalog.get("lemons")

dataset_dicts = get_sl_dicts()

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('', out.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        break

#======================================================================================================================

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("lemons",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
#
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()


#======================================================================================================================

# from detectron2.engine import DefaultPredictor
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)
#
# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_sl_dicts()
# for d in random.sample(dataset_dicts, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=balloon_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('', out.get_image()[:, :, ::-1])
#     if cv2.waitKey(0) == 27:
#         break