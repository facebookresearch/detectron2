"""
Do inference on msgpack file generated in Command Center
"""

import argparse
import io
import os

import msgpack
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class FileDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.index = []
        pos = 0
        self.file_path = file_path
        with open(file_path, "rb") as f:
            unpacker = msgpack.Unpacker(f)
            for _ in unpacker:
                self.index.append(pos)
                pos = unpacker.tell()

    def __getitem__(self, index: int):
        with open(self.file_path, "rb") as f:
            f.seek(self.index[index])
            unpacker = msgpack.Unpacker(f)
            unpacked = unpacker.unpack()
            image = Image.open(io.BytesIO(unpacked.get("data")))
            return torch.tensor(image)

    def __len__(self) -> int:
        return len(self.index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--skip_head", default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--write_original", action="store_true")
    parser.add_argument("--write_processed", action="store_true")
    parser.add_argument("--write_result_json", action="store_true")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    predictor = DefaultPredictor(cfg)
    dataset = FileDataset(args.file)

    dataloader = DataLoader(dataset, num_workers=args.num_workers)

    for i, batch in dataloader:
        print(f"Processing batch {i}")

        output = predictor(batch)
