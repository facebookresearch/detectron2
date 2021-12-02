import os
import pathlib
import sys
import supervisely_lib as sly
from detectron2.engine import DefaultPredictor
from supervisely_lib.io.fs import get_file_name_with_ext
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pathlib import Path


root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

#from nn_utils import construct_model_meta, load_model, inference

my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None

model_name_to_url_COCO = {'R50-C4(1x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl',
                     'R50-DC5(1x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl',
                     'R50-FPN(1x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl',
                     'R50-C4(3x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl',
                     'R50-DC5(3x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl',
                     'R50-FPN(3x)': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
                     'R101-C4': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl',
                     'R101-DC5': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl',
                     'R101-FPN': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl',
                     'X101-FPN': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'}


model_name_to_config_COCO = {'R50-C4(1x)': 'mask_rcnn_R_50_C4_1x.yaml',
                     'R50-DC5(1x)': 'mask_rcnn_R_50_DC5_1x.yaml',
                     'R50-FPN(1x)': 'mask_rcnn_R_50_FPN_1x.yaml',
                     'R50-C4(3x)': 'mask_rcnn_R_50_C4_3x.yaml',
                     'R50-DC5(3x)': 'mask_rcnn_R_50_DC5_3x.yaml',
                     'R50-FPN(3x)': 'mask_rcnn_R_50_FPN_3x.yaml',
                     'R101-C4': 'mask_rcnn_R_101_C4_3x.yaml',
                     'R101-DC5': 'mask_rcnn_R_101_DC5_3x.yaml',
                     'R101-FPN': 'mask_rcnn_R_101_FPN_3x.yaml',
                     'X101-FPN': 'mask_rcnn_X_101_32x8d_FPN_3x.yaml'}

model_name_to_url_LVIS = {'R50-FPN': 'https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl',
                     'R101-FPN': 'https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/model_final_824ab5.pkl',
                     'X101-FPN': 'https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x/144219108/model_final_5e3439.pkl'}

model_name_to_config_LVIS = {'R50-FPN': 'mask_rcnn_R_50_FPN_1x.yaml',
                     'R101-FPN': 'mask_rcnn_R_101_FPN_1x.yaml',
                     'X101-FPN': 'mask_rcnn_X_101_32x8d_FPN_1x.yaml'}

modelWeightsOptions = os.environ['modal.state.modelWeightsOptions']
curr_dataset = os.environ.get('modal.state.dataset', None)
pretrained_weights = os.environ['modal.state.selectedModel'] #.lower()
custom_weights = os.environ['modal.state.weightsPath']

if curr_dataset == 'COCO':
    curr_model_url = model_name_to_url_COCO[pretrained_weights]
    par_folder = 'COCO-InstanceSegmentation'
    model_config = os.path.join(par_folder, model_name_to_config_COCO[pretrained_weights])

elif curr_dataset == 'LVIS':
    curr_model_url = model_name_to_url_LVIS[pretrained_weights]
    par_folder = 'LVISv1-InstanceSegmentation'
    model_config = os.path.join(par_folder, model_name_to_config_LVIS[pretrained_weights])

else:
    raise ValueError('Choose dataset to RUN')

curr_model_name = get_file_name_with_ext(curr_model_url)

CONFIDENCE = "confidence"


# DEVICE_STR = os.environ['modal.state.device']
# final_weights = None
# model = None
# half = None
# device = None
# imgsz = None
# stride = None


# settings_path = os.path.join(root_source_path, "supervisely/serve/custom_settings.yaml")
# sly.logger.info(f"Custom inference settings path: {settings_path}")
# with open(settings_path, 'r') as file:
#     default_settings_str = file.read()
#     default_settings = yaml.safe_load(default_settings_str)


# @my_app.callback("get_output_classes_and_tags")
# @sly.timeit
# def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data=meta.to_json())
#
#
# @my_app.callback("get_session_info")
# @sly.timeit
# def get_session_info(api: sly.Api, task_id, context, state, app_logger):
#     info = {
#         "app": "YOLOv5 serve",
#         "weights": final_weights,
#         "device": str(device),
#         "half": str(half),
#         "input_size": imgsz,
#         "session_id": task_id,
#         "classes_count": len(meta.obj_classes),
#         "tags_count": len(meta.tag_metas),
#     }
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data=info)
#
#
# @my_app.callback("get_custom_inference_settings")
# @sly.timeit
# def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data={"settings": default_settings_str})
#
#
# def inference_image_path(image_path, context, state, app_logger):
#     app_logger.debug("Input path", extra={"path": image_path})
#
#     rect = None
#     if "rectangle" in state:
#         top, left, bottom, right = state["rectangle"]
#         rect = sly.Rectangle(top, left, bottom, right)
#
#     settings = state.get("settings", {})
#     for key, value in default_settings.items():
#         if key not in settings:
#             app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))
#     debug_visualization = settings.get("debug_visualization", default_settings["debug_visualization"])
#     conf_thres = settings.get("conf_thres", default_settings["conf_thres"])
#     iou_thres = settings.get("iou_thres", default_settings["iou_thres"])
#     augment = settings.get("augment", default_settings["augment"])
#
#     image = sly.image.read(image_path)  # RGB image
#     if rect is not None:
#         canvas_rect = sly.Rectangle.from_size(image.shape[:2])
#         results = rect.crop(canvas_rect)
#         if len(results) != 1:
#             return {
#                 "message": "roi rectangle out of image bounds",
#                 "roi": state["rectangle"],
#                 "img_size": {"height": image.shape[0], "width": image.shape[1]}
#             }
#         rect = results[0]
#         image = sly.image.crop(image, rect)
#     ann_json = inference(model, half, device, imgsz, stride, image, meta,
#                          conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
#                          debug_visualization=debug_visualization)
#     return ann_json
#
#
# @my_app.callback("inference_image_url")
# @sly.timeit
# def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
#     app_logger.debug("Input data", extra={"state": state})
#
#     image_url = state["image_url"]
#     ext = sly.fs.get_file_ext(image_url)
#     if ext == "":
#         ext = ".jpg"
#     local_image_path = os.path.join(my_app.data_dir, sly.rand_str(15) + ext)
#
#     sly.fs.download(image_url, local_image_path)
#     ann_json = inference_image_path(local_image_path, context, state, app_logger)
#     sly.fs.silent_remove(local_image_path)
#
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data=ann_json)
#
#
# @my_app.callback("inference_image_id")
# @sly.timeit
# def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
#     app_logger.debug("Input data", extra={"state": state})
#     image_id = state["image_id"]
#     image_info = api.image.get_info_by_id(image_id)
#     image_path = os.path.join(my_app.data_dir, sly.rand_str(10) + image_info.name)
#     api.image.download_path(image_id, image_path)
#     ann_json = inference_image_path(image_path, context, state, app_logger)
#     sly.fs.silent_remove(image_path)
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data=ann_json)
#
#
# @my_app.callback("inference_batch_ids")
# @sly.timeit
# def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
#     app_logger.debug("Input data", extra={"state": state})
#     ids = state["batch_ids"]
#     infos = api.image.get_info_by_id_batch(ids)
#     paths = []
#     for info in infos:
#         paths.append(os.path.join(my_app.data_dir, sly.rand_str(10) + info.name))
#     api.image.download_paths(infos[0].dataset_id, ids, paths)
#
#     results = []
#     for image_path in paths:
#         ann_json = inference_image_path(image_path, context, state, app_logger)
#         results.append(ann_json)
#         sly.fs.silent_remove(image_path)
#
#     request_id = context["request_id"]
#     my_app.send_response(request_id, data=results)
#
#
# def debug_inference():
#     image = sly.image.read("./data/images/bus.jpg")  # RGB
#     ann = inference(model, half, device, imgsz, image, meta, debug_visualization=True)
#     print(json.dumps(ann, indent=4))

def construct_model_meta(predictor):
    names = predictor.metadata.thing_classes

    colors = None
    if hasattr(predictor.metadata, 'thing_colors'):
        colors = predictor.metadata.thing_colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta(CONFIDENCE, sly.TagValueType.ANY_NUMBER)]

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))
    return meta


#@my_app.callback("preprocess")
@sly.timeit
def preprocess():
    # global model, half, device, imgsz, meta, final_weights
    # global stride

    # download weights
    progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
    local_path = os.path.join(my_app.data_dir, curr_model_name)

    if modelWeightsOptions == "pretrained":
        sly.fs.download(curr_model_url, local_path, my_app.cache, progress)  # TODO
    elif modelWeightsOptions == "custom":
        final_weights = custom_weights
        configs = os.path.join(Path(custom_weights).parents[1], 'opt.yaml')
        configs_local_path = os.path.join(my_app.data_dir, 'opt.yaml')
        file_info = my_app.public_api.file.get_info_by_path(TEAM_ID, custom_weights)
        progress.set(current=0, total=file_info.sizeb)
        my_app.public_api.file.download(TEAM_ID, custom_weights, local_path, my_app.cache, progress.iters_done_report)
        my_app.public_api.file.download(TEAM_ID, configs, configs_local_path)
    else:
         raise ValueError("Unknown weights option {!r}".format(modelWeightsOptions))

    # load model on device
    # model, half, device, imgsz, stride = load_model(local_path, device=DEVICE_STR)
    # meta = construct_model_meta(model)
    # sly.logger.info("Model has been successfully deployed")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = local_path


    predictor = DefaultPredictor(cfg)
    meta = construct_model_meta(predictor)
    sly.logger.info("Model has been successfully deployed")


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": TEAM_ID,
        "context.workspaceId": WORKSPACE_ID,
        "modal.state.modelWeightsOptions": modelWeightsOptions,
        "modal.state.dataset": curr_dataset,
        "modal.state.modelSize": pretrained_weights,
        "modal.state.weightsPath": custom_weights
    })

    preprocess()
    #my_app.run(initial_events=[{"command": "preprocess"}])
    my_app.run()


#@TODO: move inference methods to SDK
#@TODO: augment inference
#@TODO: https://pypi.org/project/cachetools/
if __name__ == "__main__":
    sly.main_wrapper("main", main)