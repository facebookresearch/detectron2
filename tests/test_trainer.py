import unittest
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

class TestTrainer(unittest.TestCase):
    def test_log_period(self):
        cfg = get_cfg()
        cfg.TRAINER.LOG_PERIOD = 1

        # Add minimum required config for trainer initialization
        cfg.DATASETS.TRAIN = ("dummy_dataset",)
        cfg.MODEL.DEVICE = "cpu"
        cfg.OUTPUT_DIR = "."



        def dummy_dataset():
            return [{
                "file_name": "dummy.jpg",
                "height": 500,
                "width": 500,
                "image_id": 1,
                "annotations": [{
                    "bbox": [0, 0, 10, 10],
                    "bbox_mode": 0,  # XYXY_ABS
                    "category_id": 0,
                    "iscrowd": 0,
                }]
            }]

        # Register a dummy dataset with one sample
        DatasetCatalog.register("dummy_dataset", dummy_dataset)
        MetadataCatalog.get("dummy_dataset").set(thing_classes=["dummy"])

        trainer = DefaultTrainer(cfg)
        hooks = trainer.build_hooks()
        writer_hook = hooks[-1]  # PeriodicWriter should be last hook
        self.assertEqual(writer_hook._period, 1, "Log period from config not properly set")  # Changed period to _period