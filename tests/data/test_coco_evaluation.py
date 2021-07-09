# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import json
import numpy as np
import os
import tempfile
import unittest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, Instances


class TestCOCOeval(unittest.TestCase):
    def test_fast_eval(self):
        # A small set of images/categories from COCO val
        # fmt: off
        detections = [{"image_id": 139, "category_id": 1, "bbox": [417.3332824707031, 159.27003479003906, 47.66064453125, 143.00193786621094], "score": 0.9949821829795837, "segmentation": {"size": [426, 640], "counts": "Tc`52W=3N0N4aNN^E7]:4XE1g:8kDMT;U100000001O1gE[Nk8h1dFiNY9Z1aFkN]9g2J3NdN`FlN`9S1cFRN07]9g1bFoM6;X9c1cFoM=8R9g1bFQN>3U9Y30O01OO1O001N2O1N1O4L4L5UNoE3V:CVF6Q:@YF9l9@ZF<k9[O`F=];HYnX2"}}, {"image_id": 139, "category_id": 1, "bbox": [383.5909118652344, 172.0777587890625, 17.959075927734375, 36.94813537597656], "score": 0.7685421705245972, "segmentation": {"size": [426, 640], "counts": "lZP5m0Z<300O100O100000001O00]OlC0T<OnCOT<OnCNX<JnC2bQT3"}}, {"image_id": 139, "category_id": 1, "bbox": [457.8359069824219, 158.88027954101562, 9.89764404296875, 8.771820068359375], "score": 0.07092753797769547, "segmentation": {"size": [426, 640], "counts": "bSo54T=2N2O1001O006ImiW2"}}] # noqa
        gt_annotations = {"categories": [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "furniture", "id": 65, "name": "bed"}], "images": [{"license": 4, "file_name": "000000000285.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000285.jpg", "height": 640, "width": 586, "date_captured": "2013-11-18 13:09:47", "flickr_url": "http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg", "id": 285}, {"license": 2, "file_name": "000000000139.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000139.jpg", "height": 426, "width": 640, "date_captured": "2013-11-21 01:34:01", "flickr_url": "http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg", "id": 139}], "annotations": [{"segmentation": [[428.19, 219.47, 430.94, 209.57, 430.39, 210.12, 421.32, 216.17, 412.8, 217.27, 413.9, 214.24, 422.42, 211.22, 429.29, 201.6, 430.67, 181.8, 430.12, 175.2, 427.09, 168.06, 426.27, 164.21, 430.94, 159.26, 440.29, 157.61, 446.06, 163.93, 448.53, 168.06, 448.53, 173.01, 449.08, 174.93, 454.03, 185.1, 455.41, 188.4, 458.43, 195.0, 460.08, 210.94, 462.28, 226.61, 460.91, 233.76, 454.31, 234.04, 460.08, 256.85, 462.56, 268.13, 465.58, 290.67, 465.85, 293.14, 463.38, 295.62, 452.66, 295.34, 448.26, 294.52, 443.59, 282.7, 446.06, 235.14, 446.34, 230.19, 438.09, 232.39, 438.09, 221.67, 434.24, 221.12, 427.09, 219.74]], "area": 2913.1103999999987, "iscrowd": 0, "image_id": 139, "bbox": [412.8, 157.61, 53.05, 138.01], "category_id": 1, "id": 230831}, {"segmentation": [[384.98, 206.58, 384.43, 199.98, 385.25, 193.66, 385.25, 190.08, 387.18, 185.13, 387.18, 182.93, 386.08, 181.01, 385.25, 178.81, 385.25, 175.79, 388.0, 172.76, 394.88, 172.21, 398.72, 173.31, 399.27, 176.06, 399.55, 183.48, 397.9, 185.68, 395.15, 188.98, 396.8, 193.38, 398.45, 194.48, 399.0, 205.75, 395.43, 207.95, 388.83, 206.03]], "area": 435.1449499999997, "iscrowd": 0, "image_id": 139, "bbox": [384.43, 172.21, 15.12, 35.74], "category_id": 1, "id": 233201}]} # noqa
        # fmt: on

        # Test a small dataset for typical COCO format
        experiments = {"full": (detections, gt_annotations, {})}

        # Test what happens if the list of detections or ground truth annotations is empty
        experiments["empty_dt"] = ([], gt_annotations, {})
        gt = copy.deepcopy(gt_annotations)
        gt["annotations"] = []
        experiments["empty_gt"] = (detections, gt, {})

        # Test changing parameter settings
        experiments["no_categories"] = (detections, gt_annotations, {"useCats": 0})
        experiments["no_ious"] = (detections, gt_annotations, {"iouThrs": []})
        experiments["no_rec_thrs"] = (detections, gt_annotations, {"recThrs": []})
        experiments["no_max_dets"] = (detections, gt_annotations, {"maxDets": []})
        experiments["one_max_det"] = (detections, gt_annotations, {"maxDets": [1]})
        experiments["no_area"] = (detections, gt_annotations, {"areaRng": [], "areaRngLbl": []})

        # Test what happens if one omits different fields from the annotation structure
        annotation_fields = [
            "id",
            "image_id",
            "category_id",
            "score",
            "area",
            "iscrowd",
            "ignore",
            "bbox",
            "segmentation",
        ]
        for a in annotation_fields:
            gt = copy.deepcopy(gt_annotations)
            for g in gt["annotations"]:
                if a in g:
                    del g[a]
            dt = copy.deepcopy(detections)
            for d in dt:
                if a in d:
                    del d[a]
            experiments["omit_gt_" + a] = (detections, gt, {})
            experiments["omit_dt_" + a] = (dt, gt_annotations, {})

        # Compare precision/recall for original COCO PythonAPI to custom optimized one
        for name, (dt, gt, params) in experiments.items():
            # Dump to json.
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    json_file_name = os.path.join(tmpdir, "gt_" + name + ".json")
                    with open(json_file_name, "w") as f:
                        json.dump(gt, f)
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_api = COCO(json_file_name)
            except Exception:
                pass

            for iou_type in ["bbox", "segm", "keypoints"]:
                # Run original COCOeval PythonAPI
                api_exception = None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_dt = coco_api.loadRes(dt)
                        coco_eval = COCOeval(coco_api, coco_dt, iou_type)
                        for p, v in params.items():
                            setattr(coco_eval.params, p, v)
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize()
                except Exception as ex:
                    api_exception = ex

                # Run optimized COCOeval_opt API
                opt_exception = None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_dt = coco_api.loadRes(dt)
                        coco_eval_opt = COCOeval_opt(coco_api, coco_dt, iou_type)
                        for p, v in params.items():
                            setattr(coco_eval_opt.params, p, v)
                        coco_eval_opt.evaluate()
                        coco_eval_opt.accumulate()
                        coco_eval_opt.summarize()
                except Exception as ex:
                    opt_exception = ex

                if api_exception is not None and opt_exception is not None:
                    # Original API and optimized API should throw the same exception if annotation
                    # format is bad
                    api_error = "" if api_exception is None else type(api_exception).__name__
                    opt_error = "" if opt_exception is None else type(opt_exception).__name__
                    msg = "%s: comparing COCO APIs, '%s' != '%s'" % (name, api_error, opt_error)
                    self.assertTrue(api_error == opt_error, msg=msg)
                else:
                    # Original API and optimized API should produce the same precision/recalls
                    for k in ["precision", "recall"]:
                        diff = np.abs(coco_eval.eval[k] - coco_eval_opt.eval[k])
                        abs_diff = np.max(diff) if diff.size > 0 else 0.0
                        msg = "%s: comparing COCO APIs, %s differs by %f" % (name, k, abs_diff)
                        self.assertTrue(abs_diff < 1e-4, msg=msg)

    @unittest.skipIf(os.environ.get("CI"), "Require COCO data.")
    def test_unknown_category(self):
        dataset = "coco_2017_val_100"
        evaluator = COCOEvaluator(dataset)
        evaluator.reset()
        inputs = DatasetCatalog.get(dataset)[:2]
        pred = Instances((100, 100))
        pred.pred_boxes = Boxes(torch.rand(2, 4))
        pred.scores = torch.rand(2)
        pred.pred_classes = torch.tensor([10, 80])
        output = {"instances": pred}
        evaluator.process(inputs, [output, output])
        with self.assertRaises(AssertionError):
            evaluator.evaluate()
