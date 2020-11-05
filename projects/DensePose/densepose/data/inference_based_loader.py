# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple
import torch
from torch import nn

SampledData = Any
ModelOutput = Any


def _grouper(iterable: Iterable[Any], n: int, fillvalue=None) -> Iterator[Tuple[Any]]:
    """
    Group elements of an iterable by chunks of size `n`, e.g.
    grouper(range(9), 4) ->
        (0, 1, 2, 3), (4, 5, 6, 7), (8, None, None, None)
    """
    it = iter(iterable)
    while True:
        values = []
        for _ in range(n):
            try:
                value = next(it)
            except StopIteration:
                if values:
                    values.extend([fillvalue] * (n - len(values)))
                    yield tuple(values)
                return
            values.append(value)
        yield tuple(values)


class ScoreBasedFilter:
    """
    Filters entries in model output based on their scores
    Discards all entries with score less than the specified minimum
    """

    def __init__(self, min_score: float = 0.8):
        self.min_score = min_score

    def __call__(self, model_output: ModelOutput) -> ModelOutput:
        for model_output_i in model_output:
            instances = model_output_i["instances"]
            if not instances.has("scores"):
                continue
            instances_filtered = instances[instances.scores >= self.min_score]
            model_output_i["instances"] = instances_filtered
        return model_output


class InferenceBasedLoader:
    """
    Data loader based on results inferred by a model. Consists of:
     - a data loader that provides batches of images
     - a model that is used to infer the results
     - a data sampler that converts inferred results to annotations
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable[List[torch.Tensor]],
        data_sampler: Optional[Callable[[ModelOutput], List[SampledData]]] = None,
        data_filter: Optional[Callable[[ModelOutput], ModelOutput]] = None,
        shuffle: bool = True,
        batch_size: int = 4,
        inference_batch_size: int = 4,
        drop_last: bool = False,
    ):
        """
        Constructor

        Args:
          model (torch.nn.Module): model used to produce data
          data_loader (Iterable[Tensor]): iterable that provides images
              to perform inference on
          data_sampler (Callable: ModelOutput -> SampledData): functor
              that produces annotation data from inference results;
              (optional, default: None)
          data_filter (Callable: ModelOutput -> ModelOutput): filter
              that selects model outputs for for further processing
              (optional, default: None)
          shuffle (bool): if True, the input images get shuffled
          batch_size (int): batch size for the produced annotation data
          inference_batch_size (int): batch size for input images
          drop_last (bool): if True, drop the last batch if it is undersized
        """
        self.model = model
        self.model.eval()
        self.data_loader = data_loader
        self.data_sampler = data_sampler
        self.data_filter = data_filter
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[SampledData]]:
        for batch in self.data_loader:
            # batch : List[Tensor[N, C, H, W]]
            # images_batch : Tensor[N, C, H, W]
            # image : Tensor[C, H, W]
            images = [image for images_batch in batch for image in images_batch]
            if not images:
                continue
            if self.shuffle:
                random.shuffle(images)
            yield from self._produce_data(images)

    def _produce_data(self, images: List[torch.Tensor]) -> Iterator[List[SampledData]]:
        """
        Produce batches of data from images

        Args:
          images (List[Tensor]): list of images to process

        Returns:
          Iterator over batches of data sampled from model outputs
        """
        data_batches: List[SampledData] = []
        batched_images = _grouper(images, self.inference_batch_size)
        for batch in batched_images:
            batch = [{"image": img.to(self.model.device)} for img in batch if img is not None]
            if not batch:
                continue
            with torch.no_grad():
                model_output = self.model(batch)
            for model_output_i, batch_i in zip(model_output, batch):
                model_output_i["image"] = batch_i["image"]
            model_output_filtered = (
                model_output if self.data_filter is None else self.data_filter(model_output)
            )
            data = (
                model_output_filtered
                if self.data_sampler is None
                else self.data_sampler(model_output_filtered)
            )
            for data_i in data:
                if len(data_i["instances"]):
                    data_batches.append(data_i)
            if len(data_batches) >= self.batch_size:
                yield data_batches[: self.batch_size]
                data_batches = data_batches[self.batch_size :]
        if not self.drop_last and data_batches:
            yield data_batches
