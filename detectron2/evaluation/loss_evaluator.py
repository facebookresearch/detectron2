from typing import Dict

from detectron2.engine import SimpleTrainer
from detectron2.evaluation import DatasetEvaluator


class LossEvaluator(DatasetEvaluator):
    """
    Write validation loss to tensorboard.
    Assumes that the output of the model arc is a tuple (model_out, losses)
    """
    VAL_PREFIX = "Validation_"
    LOSSES_DICT_STR = "losses_dict"

    def __init__(self):
        self._aggregated_losses = {}
        self._num_datapoints = 0

    def process(self, inputs, outputs: Dict):
        """
        Add calculated loss to aggregation dict, assume that all losses always exist
        """
        # in case this evaluator is invoked in inference
        if not isinstance(outputs, tuple):
            return

        assert len(outputs) == 2

        losses_dict = outputs[1]
        if not isinstance(losses_dict, dict):
            losses_dict = {'total_loss': losses_dict}

        for loss_name, loss_value in losses_dict.items():
            validation_loss_name = LossEvaluator.VAL_PREFIX + loss_name
            loss_value_cpu = loss_value.cpu().detach()

            if self._num_datapoints == 0:
                self._aggregated_losses[validation_loss_name] = loss_value_cpu
            else:
                assert validation_loss_name in self._aggregated_losses, (f"Adding new losses on\
                    the fly is not supported, loss {validation_loss_name} \
                    appeared on datapoint {self._num_datapoints}")
                self._aggregated_losses[validation_loss_name] = (
                    self._num_datapoints * self._aggregated_losses[validation_loss_name]
                    + loss_value_cpu) / (self._num_datapoints + 1)
        self._num_datapoints += 1

    def evaluate(self):
        try:
            SimpleTrainer.write_metrics(
                self._aggregated_losses, 0, LossEvaluator.VAL_PREFIX
            )
        except AssertionError as e:
            # If there is no storage context, this method will not work, otherwise, raise exception
            if not e.args[0].find("EventStorage") >= 0:
                raise
