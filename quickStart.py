import torch, torchvision
torch.cuda.empty_cache()
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
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)
import argparse
import time

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


'''
Getting predefined Configurations and return edited configuration
'''

def get_config(config):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ("test", )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.RETINANET.NUM_CLASSES = 5
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.9
    cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.OUTPUT_DIR = "output"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.TEST.EVAL_PERIOD = 2000 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    return cfg


'''
Getting Training Model from CocoTrainer()
'''

def get_model(config):
    cfg = get_config(config)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg) 
    return trainer


'''
Prediction function on one image
'''
def predict(image_path, metadata, config):
    cfg = get_config(config)
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.OUTPUT_DIR = "torchscript"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "deploy.pt")
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(image_path)
    time1 = time.time()
    output = predictor(im)
    print(output)
    time2 = time.time()
    print("Inference Time:", time2-time1)
    v = Visualizer(im[:, :, ::-1],
                    metadata,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imwrite("output.jpg", out.get_image()[:,:,::-1])



if __name__ == '__main__':
    #Getting Argument from Command Line
    parser = argparse.ArgumentParser(description='Template Detection')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to Config File')
    parser.add_argument('-train', '--train', type=str, help='Path to Train Dataset')
    parser.add_argument('-test', '--test', type=str, required=True, help='Path to Test Dataset')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Training/Inference')
    parser.add_argument('-i', '--image', type=str, help='Path to image for inference')
    args = parser.parse_args()
    '''
    python detectron2.py -c <path of config file> 
                         -m <train/test> 
                         -train <path to training data directory> 
                         -test <path to validation data directory> 
                         -i <test image>
    '''


    
    #Register the Trainig Dataset 
    register_coco_instances("test",{},f"{args.test}/_annotations.coco.json",f"{args.test}")
    train_metadata = MetadataCatalog.get("test")
    dataset_dicts = DatasetCatalog.get("test")


    if (args.mode == 'train'):
        register_coco_instances("train",{},f"{args.train}/_annotations.coco.json",f"{args.train}")
        trainer = get_model(args.config)
        trainer.resume_or_load(resume=True)
        trainer.train()
        # export_onxx(cfg, model)
    elif (args.mode == 'test'):
        predict(f"{args.image}", train_metadata, args.config)
 
