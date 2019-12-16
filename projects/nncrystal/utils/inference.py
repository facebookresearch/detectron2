import io
from typing import List

import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils import msgpack_data_index, load_images_entries


class BatchPredictor(DefaultPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def __call__(self, original_images: List):
        """
        Args:
            original_image (List[np.ndarray]): images of shape (H, W, C) (in BGR order).

        Returns:
            predictions (List[dict]): the output of the model
        """
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_images = [x[:, :, ::-1] for x in original_images]
        height, width = original_images[0].shape[:2]
        images = [self.transform_gen.get_transform(x).apply_image(x) for x in original_images]
        images = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1)) for x in images]

        inputs = [{"image": x, "height": height, "width": width} for x in images]
        predictions = self.model(inputs)
        return predictions


class MsgpackFileDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.index = []
        pos = 0
        self.file_path = file_path
        self.index = msgpack_data_index(file_path)

    def __getitem__(self, index: int):
        entry = next(load_images_entries(self.file_path, [index], self.index))
        name = entry[b"name"]
        data = entry[b"data"]
        image = np.asarray(Image.open(io.BytesIO(data)))
        image = np.stack((image,) * 3, axis=-1)

        return name, image

    def __len__(self) -> int:
        return len(self.index)


def get_data_loader(dataset, num_workers=4, batch_size=8, collate_fn=None):
    if collate_fn is None:
        def collate_fn(batch):
            names = [x[0] for x in batch]
            images = [x[1] for x in batch]
            return names, images
    return DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=batch_size)


def get_predictor(config, weights=None):
    cfg = get_cfg()
    cfg.merge_from_file(config)
    if weights:
        cfg.MODEL.WEIGHTS = weights

    predictor = BatchPredictor(cfg)
    return predictor


def batch_inference_result_split(results: List, keys: List):
    """
    Split inference result into a dict with provided key and result dicts.
    The length of results and keys must match
    :param results:
    :param keys:
    :return:
    """

    ret = {}
    for result, key in zip(results, keys):
        ret[key] = result

    return ret
