# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import time
from typing import List, Mapping, Optional
import torch

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.engine import TrainerBase
import copy


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, models, data_loader, optimizer, strong_aug=None, warm_up_iter=None, crt_on=False):
        """
        Args:
            model: a Dict of torch Module. Takes a data from data_loader and returns a
                dict of losses. {str: nn.Module}
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        student_model = models["student"]
        teacher_model = models["teacher"]
        student_model.train()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.roi_heads.it_is_teacher()
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.strong_aug = strong_aug
        # self.corrector = corrector
        # self.corrector.eval()
        # for param in corrector.parameters():
        #     param.requires_grad = False
        self.warm_up_iter = warm_up_iter
        self.crt_on = crt_on

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.student_model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        Firstly, Use teacher model to get pseudo labels
        """
        teacher_input = [x["instances"] for x in data]
        for x in teacher_input:
            x.pred_boxes = x.gt_boxes
            x.pred_classes = x.gt_classes

        teacher_output = self.teacher_model.inference(copy.deepcopy(data), teacher_input, do_postprocess=False)
        # for t in teacher_output:
        #     t.pred_densepose.u = (t.pred_densepose.u * 255.0).clamp(0, 255.0) / 255.0
        #     t.pred_densepose.v = (t.pred_densepose.v * 255.0).clamp(0, 255.0) / 255.0

        del teacher_input

        # crt_loss = None
        # if self.corrector is not None:
        #     correction, crt_loss = self.corrector(teacher_output)
        #     self.corrector.correct(correction, teacher_output, self.iter)

        # Construct input data for student model
        for i, x in enumerate(data):
            pseudo_label = teacher_output[i].pred_densepose
            # pseudo_segm = torch.argmax(pseudo_label.fine_segm, dim=1).long()
            pseudo_segm = pseudo_label.fine_segm
            # == baseline ==
            pseudo_u = pseudo_label.u
            pseudo_v = pseudo_label.v

            x["cur_iter"] = self.iter

            # == block ==
            # pseudo_u = pseudo_label.u_cls
            # pseudo_v = pseudo_label.v_cls

            # == cse ==
            # pseudo_embed = pseudo_label.embedding
            for j, densepose_data in enumerate(x["instances"].gt_densepose):
                if densepose_data is not None:
                    # == densepose rcnn ==
                    densepose_data.set("dp_p_segm", pseudo_segm[j])
                    densepose_data.set("dp_p_u", pseudo_u[j])
                    densepose_data.set("dp_p_v", pseudo_v[j])

                    # == cse ==
                    # densepose_data.set("dp_p_embed", pseudo_embed[j])

        del teacher_output

        if self.strong_aug is not None:
            for aug in self.strong_aug:
                data = aug(data)

        loss_dict = self.student_model(data)
        # if crt_loss is not None:
        #     loss_dict.update(crt_loss)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            # if self.crt_on:
            #     factor = np.exp(-5 * (1 - self.iter / self.warm_up_iter) ** 2)
            #     loss_dict["loss_unsup_segm"] = loss_dict["loss_unsup_segm"] * factor
            #     loss_dict["loss_u_p"] = loss_dict["loss_u_p"] * factor
            #     loss_dict["loss_v_p"] = loss_dict["loss_v_p"] * factor
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])
