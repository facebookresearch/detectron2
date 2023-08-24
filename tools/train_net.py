#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("SODA_train",)
    cfg.DATASETS.TEST = ("SODA_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

from detectron2.utils.events import get_event_storage
from torch import cat
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

from collections import defaultdict
import random
class TsneCal(hooks.HookBase):
    def __init__(self, eval_period):
        self._period = eval_period

    def after_step(self):
        # if not self._vis_tsne: #
        #     return
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            storage = get_event_storage()
            if len(storage._vis_tsne_feature["feature"]) == 0:
                assert 1==2, "no feature in storage"
            else:
                features = cat(storage._vis_tsne_feature["feature"], dim=0)
                labels = cat(storage._vis_tsne_feature["label"], dim=0)

                torch.save(features, "tsne_feautures{}.pth".format(self.trainer.iter))
                torch.save(labels, "tsne_labels{}.pth".format(self.trainer.iter))

                result = defaultdict(list)

                for i in range(6):
                    area = labels == i
                    cls_feature = features[area]

                    B, D = cls_feature.shape
                    Num = min(500, B)
                    index = torch.LongTensor(random.sample(range(B), Num)).cuda()

                    cls_feature = torch.index_select(cls_feature, 0, index)

                    label = torch.full([Num, ], i)

                    result["labels"].append(label)
                    result['features'].append(cls_feature)

                tsne = TSNE(n_components=2, verbose=1)

                result_l = torch.cat(result["labels"], dim=0)
                result_f = torch.cat(result['features'], dim=0)

                del storage._vis_tsne_feature['features'], storage._vis_tsne_feature['labels']

                storage._vis_tsne_feature.clear()

                result_2D = tsne.fit_transform(result_f.cpu())
                fig1 = plot_embedding_2D(result_2D, result_l, 't-SNE')
                plt.savefig("ATT_tsne{}.png".format(self.trainer.iter))




color_map = ['r','y','k','g','b','m','c'] # 7个类，准备7种颜色
def plot_embedding_2D(data, label, title):
    """

    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def main(args):
    cfg = setup(args)
    register_coco_instances("SODA_train", {}, "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_train.json",
                            "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/train/")
    register_coco_instances("SODA_val", {}, "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_val.json",
                            "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/val/")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    with torch.no_grad():
        trainer = Trainer(cfg)

        trainer.resume_or_load(resume=args.resume)
        if cfg.TEST.TSNE:
            trainer.register_hooks([TsneCal(cfg.TEST.EVAL_PERIOD)])
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks(
                [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
            )
        return trainer.train()

def my_pre():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("SODA_train", {}, "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_train.json",
                            "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/train/")
    register_coco_instances("SODA_val", {}, "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/annotations/instance_val.json",
                            "/home/jnx/code/Object_Detection/SSLAD-2D/labeled/val/")


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
