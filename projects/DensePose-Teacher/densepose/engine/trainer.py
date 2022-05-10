# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import imp
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Union
import weakref
from detectron2.engine.defaults import default_writers
import torch
from torch import nn
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.solver import build_lr_scheduler
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.engine.train_loop import TrainerBase
from detectron2.engine import create_ddp_model, hooks
from detectron2.modeling import build_model

from densepose import DensePoseDatasetMapperTTA, DensePoseGeneralizedRCNNWithTTA, load_from_cfg
from densepose.data import (
    DatasetMapper,
    build_combined_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    build_inference_based_loaders,
    has_inference_based_loaders,
)
from densepose.evaluation.d2_evaluator_adapter import Detectron2COCOEvaluatorAdapter
from densepose.evaluation.evaluator import DensePoseCOCOEvaluator, build_densepose_evaluator_storage
from densepose.modeling.cse import Embedder
from .train_loop import SimpleTrainer
from .mean_teacher import MeanTeacher


class SampleCountingLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        it = iter(self.loader)
        storage = get_event_storage()
        while True:
            try:
                batch = next(it)
                num_inst_per_dataset = {}
                for data in batch:
                    dataset_name = data["dataset"]
                    if dataset_name not in num_inst_per_dataset:
                        num_inst_per_dataset[dataset_name] = 0
                    num_inst = len(data["instances"])
                    num_inst_per_dataset[dataset_name] += num_inst
                for dataset_name in num_inst_per_dataset:
                    storage.put_scalar(f"batch/{dataset_name}", num_inst_per_dataset[dataset_name])
                yield batch
            except StopIteration:
                break


class SampleCountMetricPrinter(EventWriter):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write(self):
        storage = get_event_storage()
        batch_stats_strs = []
        for key, buf in storage.histories().items():
            if key.startswith("batch/"):
                batch_stats_strs.append(f"{key} {buf.avg(20)}")
        self.logger.info(", ".join(batch_stats_strs))


class Trainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        student_model = self.build_model(cfg)
        teacher_model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, student_model)
        data_loader = self.build_train_loader(cfg)

        student_model = create_ddp_model(student_model, broadcast_buffers=False)
        # teacher_model = create_ddp_model(teacher_model, broadcast_buffers=False)
        self._trainer = SimpleTrainer({"teacher": teacher_model, "student": student_model}, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.student_checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            student_model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.teacher_checkpointer = DetectionCheckpointer(
            teacher_model,
            cfg.MODEL.SEMI.TEACHER_OUTPUT,
            trainer=weakref.proxy(self),
        )

        if comm.is_main_process():
            if not os.path.exists(cfg.MODEL.SEMI.TEACHER_OUTPUT):
                os.mkdir(cfg.MODEL.SEMI.TEACHER_OUTPUT)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.student_checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        teacher_weights = self.cfg.MODEL.SEMI.TEACHER_WEIGHTS
        if teacher_weights is None or teacher_weights == "":
            teacher_weights = self.cfg.MODEL.WEIGHTS
        self.teacher_checkpointer.resume_or_load(teacher_weights, resume=resume)
        if resume and self.student_checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.student_model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.student_model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.student_checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(hooks.PeriodicCheckpointer(self.teacher_checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(MeanTeacher())

        def test_and_save_results():
            if cfg.MODEL.SEMI.INFERENCE_ON == "student":
                self._last_eval_results = self.test(self.cfg, self.student_model)
            elif cfg.MODEL.SEMI.INFERENCE_ON == "teacher":
                self._last_eval_results = self.test(self.cfg, self.teacher_model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        return ret

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg

    @classmethod
    def extract_embedder_from_model(cls, model: nn.Module) -> Optional[Embedder]:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "embedder"):
            return model.roi_heads.embedder
        return None

    # TODO: the only reason to copy the base class code here is to pass the embedder from
    # the model to the evaluator; that should be refactored to avoid unnecessary copy-pasting
    @classmethod
    def test(
        cls,
        cfg: CfgNode,
        model: nn.Module,
        evaluators: Optional[Union[DatasetEvaluator, List[DatasetEvaluator]]] = None,
    ):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (DatasetEvaluator, list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    embedder = cls.extract_embedder_from_model(model)
                    evaluator = cls.build_evaluator(cfg, dataset_name, embedder=embedder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE or comm.is_main_process():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            else:
                results_i = {}
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_evaluator(
        cls,
        cfg: CfgNode,
        dataset_name: str,
        output_folder: Optional[str] = None,
        embedder: Optional[Embedder] = None,
    ) -> DatasetEvaluators:
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = []
        distributed = cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE
        # Note: we currently use COCO evaluator for both COCO and LVIS datasets
        # to have compatible metrics. LVIS bbox evaluator could also be used
        # with an adapter to properly handle filtered / mapped categories
        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # if evaluator_type == "coco":
        #     evaluators.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # elif evaluator_type == "lvis":
        #     evaluators.append(LVISEvaluator(dataset_name, output_dir=output_folder))
        evaluators.append(
            Detectron2COCOEvaluatorAdapter(
                dataset_name, output_dir=output_folder, distributed=distributed
            )
        )
        if cfg.MODEL.DENSEPOSE_ON:
            storage = build_densepose_evaluator_storage(cfg, output_folder)
            evaluators.append(
                DensePoseCOCOEvaluator(
                    dataset_name,
                    distributed,
                    output_folder,
                    evaluator_type=cfg.DENSEPOSE_EVALUATION.TYPE,
                    min_iou_threshold=cfg.DENSEPOSE_EVALUATION.MIN_IOU_THRESHOLD,
                    storage=storage,
                    embedder=embedder,
                    should_evaluate_mesh_alignment=cfg.DENSEPOSE_EVALUATION.EVALUATE_MESH_ALIGNMENT,
                    mesh_alignment_mesh_names=cfg.DENSEPOSE_EVALUATION.MESH_ALIGNMENT_MESH_NAMES,
                )
            )
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "features": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.FEATURES_LR_FACTOR,
                },
                "embeddings": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_LR_FACTOR,
                },
            },
        )
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        return maybe_add_gradient_clipping(cfg, optimizer)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        if not has_inference_based_loaders(cfg):
            return data_loader
        model = cls.build_model(cfg)
        model.to(cfg.BOOTSTRAP_MODEL.DEVICE)
        DetectionCheckpointer(model).resume_or_load(cfg.BOOTSTRAP_MODEL.WEIGHTS, resume=False)
        inference_based_loaders, ratios = build_inference_based_loaders(cfg, model)
        loaders = [data_loader] + inference_based_loaders
        ratios = [1.0] + ratios
        combined_data_loader = build_combined_loader(cfg, loaders, ratios)
        sample_counting_loader = SampleCountingLoader(combined_data_loader)
        return sample_counting_loader

    def build_writers(self):
        writers = default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
        writers.append(SampleCountMetricPrinter())
        return writers

    @classmethod
    def test_with_TTA(cls, cfg: CfgNode, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        transform_data = load_from_cfg(cfg)
        model = DensePoseGeneralizedRCNNWithTTA(
            cfg, model, transform_data, DensePoseDatasetMapperTTA(cfg)
        )
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)  # pyre-ignore[6]
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

# Access basic attributes from the underlying trainer
for _attr in ["student_model", "teacher_model", "data_loader", "optimizer"]:
    setattr(
        Trainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
