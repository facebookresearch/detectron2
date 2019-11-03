from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from fvcore.common.checkpoint import Checkpointer


def evaluate_on_dataset(
        config_file="../../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        override_cfg=(),
        test_datasets=(),
):
    if override_cfg is None:
        override_cfg = []
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(override_cfg)
    cfg.DATASETS.TEST = test_datasets
    model = build_model(cfg)

    checkpointer = Checkpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]

    metrics = DefaultTrainer.test(cfg, model, evaluator)

    return metrics