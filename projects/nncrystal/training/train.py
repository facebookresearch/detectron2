import json
import os
import logging

from detectron2.evaluation import COCOEvaluator

logger = logging.getLogger("detectron2")
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False)


def train(
        config_file: str = "../../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        override_cfg=(),
        resume=True,
        weight_fallback="detectron2://ImageNetPretrained/MSRA/R-101.pkl",
        restart=False,
        force_test=False,
):
    if override_cfg is None:
        override_cfg = []
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(override_cfg)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    try:
        trainer.resume_or_load(resume=resume)
    except:
        logger.warning(f"Failed to load config file weights: {cfg.MODEL.WEIGHTS}, "
                       f"fall back to: {weight_fallback}")
        cfg.MODEL.WEIGHTS = weight_fallback
        trainer.resume_or_load(resume=False)

    if restart:
        trainer.start_iter = 0

    trainer.train()
    if force_test:
        test_output = trainer.test(cfg, trainer.model)
        with open(os.path.join(cfg.OUTPUT_DIR, "test_result.json"), "w") as f:
            json.dump(test_output, f)

