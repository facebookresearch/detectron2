from flask import Flask, jsonify, request, render_template, redirect
import torch, torchvision
from PIL import Image
from base64 import encodebytes
import io
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode
import argparse
import time

register_coco_instances("test",{},"/data/object_detection/Validation/_annotations.coco.json","/data/object_detection/Validation")

'''
COCO Trainer - Overriding the method of evaluator 
'''
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def get_config():
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ("test", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = "output"
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.RETINANET.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    return cfg


app = Flask(__name__, template_folder='Templates')


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == "POST":
        if request.files:
            cfg = get_config()
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2   # set a custom testing threshold
            predictor = DefaultPredictor(cfg)
            image = request.files["file"]
            image_byte = image.read()
            image_read = Image.open(io.BytesIO(image_byte)).convert('RGB')
            image_byte = np.array(image_read)
            image_byte = image_byte[:,:,::-1].copy()
            time1 = time.time()
            output = predictor(image_byte)
            time2 = time.time()
            print(time2-time1)
            metadata = MetadataCatalog.get("test")
            dataset_dicts = DatasetCatalog.get("test")
            class_names = metadata.thing_classes
            pred_classes = output['instances'].pred_classes.tolist()
            pred_class_names = list(map(lambda x: class_names[x], pred_classes))
            boxes = output['instances'].pred_boxes.tensor.tolist()
            score = output['instances'].scores.tolist()
            croped = list()
            for i in boxes:
                croped_image = image_read.crop(tuple(i))
                img_byte_arr = io.BytesIO()
                croped_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                img_byte_arr = encodebytes(img_byte_arr).decode('ascii')
                croped.append(img_byte_arr)
            response = {'boxes':boxes, 'scores':score, 'predicted_classes':pred_class_names, 'classes':pred_classes, 'croped_image':croped}


            
            return jsonify(response)
            
    return render_template("image_upload.html")


if __name__== '__main__':
    print("Yes")
    app.run(host='172.22.3.13',port=80)
