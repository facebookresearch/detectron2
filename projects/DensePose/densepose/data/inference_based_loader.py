# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

import random
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
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
        data_loader: Iterable[List[Dict[str, Any]]],
        data_sampler: Optional[Callable[[ModelOutput], List[SampledData]]] = None,
        data_filter: Optional[Callable[[ModelOutput], ModelOutput]] = None,
        shuffle: bool = True,
        batch_size: int = 4,
        inference_batch_size: int = 4,
        drop_last: bool = False,
        category_to_class_mapping: Optional[dict] = None,
    ):
        """
        Constructor

        Args:
          model (torch.nn.Module): model used to produce data
          data_loader (Iterable[List[Dict[str, Any]]]): iterable that provides
            dictionaries with "images" and "categories" fields to perform inference on
          data_sampler (Callable: ModelOutput -> SampledData): functor
              that produces annotation data from inference results;
              (optional, default: None)
          data_filter (Callable: ModelOutput -> ModelOutput): filter
              that selects model outputs for further processing
              (optional, default: None)
          shuffle (bool): if True, the input images get shuffled
          batch_size (int): batch size for the produced annotation data
          inference_batch_size (int): batch size for input images
          drop_last (bool): if True, drop the last batch if it is undersized
          category_to_class_mapping (dict): category to class mapping
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
        if category_to_class_mapping is not None:
            self.category_to_class_mapping = category_to_class_mapping
        else:
            self.category_to_class_mapping = {}

    def __iter__(self) -> Iterator[List[SampledData]]:
        for batch in self.data_loader:
            # batch : List[Dict[str: Tensor[N, C, H, W], str: Optional[str]]]
            # images_batch : Tensor[N, C, H, W]
            # image : Tensor[C, H, W]
            images_and_categories = [
                {"image": image, "category": category}
                for element in batch
                for image, category in zip(element["images"], element["categories"])
            ]
            if not images_and_categories:
                continue
            if self.shuffle:
                random.shuffle(images_and_categories)
            yield from self._produce_data(images_and_categories)  # pyre-ignore[6]

    def _produce_data(
        self, images_and_categories: List[Tuple[torch.Tensor, Optional[str]]]
    ) -> Iterator[List[SampledData]]:
        """
        Produce batches of data from images

        Args:
          images_and_categories (List[Tuple[torch.Tensor, Optional[str]]]):
            list of images and corresponding categories to process

        Returns:
          Iterator over batches of data sampled from model outputs
        """
        data_batches: List[SampledData] = []
        category_to_class_mapping = self.category_to_class_mapping
        batched_images_and_categories = _grouper(images_and_categories, self.inference_batch_size)
        for batch in batched_images_and_categories:
            batch = [
                {
                    "image": image_and_category["image"].to(self.model.device),
                    "category": image_and_category["category"],
                }
                for image_and_category in batch
                if image_and_category is not None
            ]
            if not batch:
                continue
            with torch.no_grad():
                model_output = self.model(batch)
            for model_output_i, batch_i in zip(model_output, batch):
                assert len(batch_i["image"].shape) == 3
                model_output_i["image"] = batch_i["image"]
                instance_class = category_to_class_mapping.get(batch_i["category"], 0)
                model_output_i["instances"].dataset_classes = torch.tensor(
                    [instance_class] * len(model_output_i["instances"])
                )
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
