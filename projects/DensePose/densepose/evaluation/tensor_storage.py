# Copyright (c) Facebook, Inc. and its affiliates.

import io
import numpy as np
import os
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import BinaryIO, Dict, Tuple
import torch

from detectron2.utils.comm import gather, get_rank
from detectron2.utils.file_io import PathManager


@dataclass
class SizeData:
    dtype: str
    shape: Tuple[int]


def _calculate_record_field_size_b(data_schema: Dict[str, SizeData], field_name: str) -> int:
    schema = data_schema[field_name]
    element_size_b = np.dtype(schema.dtype).itemsize
    record_field_size_b = reduce(mul, schema.shape) * element_size_b
    return record_field_size_b


def _calculate_record_size_b(data_schema: Dict[str, SizeData]) -> int:
    record_size_b = 0
    for field_name in data_schema:
        record_field_size_b = _calculate_record_field_size_b(data_schema, field_name)
        record_size_b += record_field_size_b
    return record_size_b


def _calculate_record_field_sizes_b(data_schema: Dict[str, SizeData]) -> Dict[str, int]:
    field_sizes_b = {}
    for field_name in data_schema:
        field_sizes_b[field_name] = _calculate_record_field_size_b(data_schema, field_name)
    return field_sizes_b


class SingleProcessTensorStorage:
    """
    Compact tensor storage to keep tensor data of predefined size and type.
    """

    def __init__(self, data_schema: Dict[str, SizeData], storage_impl: BinaryIO):
        """
        Construct tensor storage based on information on data shape and size.
        Internally uses numpy to interpret the type specification.
        The storage must support operations `seek(offset, whence=os.SEEK_SET)` and
        `read(size)` to be able to perform the `get` operation.
        The storage must support operation `write(bytes)` to be able to perform
        the `put` operation.

        Args:
            data_schema (dict: str -> SizeData): dictionary which maps tensor name
                to its size data (shape and data type), e.g.
                ```
                {
                  "coarse_segm": SizeData(dtype="float32", shape=(112, 112)),
                  "embedding": SizeData(dtype="float32", shape=(16, 112, 112)),
                }
                ```
            storage_impl (BinaryIO): io instance that handles file-like seek, read
                and write operations, e.g. a file handle or a memory buffer like io.BytesIO
        """
        self.data_schema = data_schema
        self.record_size_b = _calculate_record_size_b(data_schema)
        self.record_field_sizes_b = _calculate_record_field_sizes_b(data_schema)
        self.storage_impl = storage_impl
        self.next_record_id = 0

    def get(self, record_id: int) -> Dict[str, torch.Tensor]:
        """
        Load tensors from the storage by record ID

        Args:
            record_id (int): Record ID, for which to load the data

        Return:
            dict: str -> tensor: tensor name mapped to tensor data, recorded under the provided ID
        """
        self.storage_impl.seek(record_id * self.record_size_b, os.SEEK_SET)
        data_bytes = self.storage_impl.read(self.record_size_b)
        assert len(data_bytes) == self.record_size_b, (
            f"Expected data size {self.record_size_b} B could not be read: "
            f"got {len(data_bytes)} B"
        )
        record = {}
        cur_idx = 0
        # it's important to read and write in the same order
        for field_name in sorted(self.data_schema):
            schema = self.data_schema[field_name]
            field_size_b = self.record_field_sizes_b[field_name]
            chunk = data_bytes[cur_idx : cur_idx + field_size_b]
            data_np = np.frombuffer(
                chunk, dtype=schema.dtype, count=reduce(mul, schema.shape)
            ).reshape(schema.shape)
            record[field_name] = torch.from_numpy(data_np)
            cur_idx += field_size_b
        return record

    def put(self, data: Dict[str, torch.Tensor]) -> int:
        """
        Store tensors in the storage

        Args:
            data (dict: str -> tensor): data to store, a dictionary which maps
                tensor names into tensors; tensor shapes must match those specified
                in data schema.
        Return:
            int: record ID, under which the data is stored
        """
        # it's important to read and write in the same order
        for field_name in sorted(self.data_schema):
            assert (
                field_name in data
            ), f"Field '{field_name}' not present in data: data keys are {data.keys()}"
            value = data[field_name]
            assert value.shape == self.data_schema[field_name].shape, (
                f"Mismatched tensor shapes for field '{field_name}': "
                f"expected {self.data_schema[field_name].shape}, got {value.shape}"
            )
            data_bytes = value.cpu().numpy().tobytes()
            assert len(data_bytes) == self.record_field_sizes_b[field_name], (
                f"Expected field {field_name} to be of size "
                f"{self.record_field_sizes_b[field_name]} B, got {len(data_bytes)} B"
            )
            self.storage_impl.write(data_bytes)
        record_id = self.next_record_id
        self.next_record_id += 1
        return record_id


class SingleProcessFileTensorStorage(SingleProcessTensorStorage):
    """
    Implementation of a single process tensor storage which stores data in a file
    """

    def __init__(self, data_schema: Dict[str, SizeData], fpath: str, mode: str):
        self.fpath = fpath
        assert "b" in mode, f"Tensor storage should be opened in binary mode, got '{mode}'"
        if "w" in mode:
            file_h = PathManager.open(fpath, mode)
        elif "r" in mode:
            local_fpath = PathManager.get_local_path(fpath)
            file_h = open(local_fpath, mode)
        else:
            raise ValueError(f"Unsupported file mode {mode}, supported modes: rb, wb")
        super().__init__(data_schema, file_h)


class SingleProcessRamTensorStorage(SingleProcessTensorStorage):
    """
    Implementation of a single process tensor storage which stores data in RAM
    """

    def __init__(self, data_schema: Dict[str, SizeData], buf: io.BytesIO):
        super().__init__(data_schema, buf)


class MultiProcessTensorStorage:
    """
    Representation of a set of tensor storages created by individual processes,
    allows to access those storages from a single owner process. The storages
    should either be shared or broadcasted to the owner process.
    The processes are identified by their rank, data is uniquely defined by
    the rank of the process and the record ID.
    """

    def __init__(self, rank_to_storage: Dict[int, SingleProcessTensorStorage]):
        self.rank_to_storage = rank_to_storage

    def get(self, rank: int, record_id: int) -> Dict[str, torch.Tensor]:
        storage = self.rank_to_storage[rank]
        return storage.get(record_id)

    def put(self, rank: int, data: Dict[str, torch.Tensor]) -> int:
        storage = self.rank_to_storage[rank]
        return storage.put(data)


class MultiProcessFileTensorStorage(MultiProcessTensorStorage):
    def __init__(self, data_schema: Dict[str, SizeData], rank_to_fpath: Dict[int, str], mode: str):
        rank_to_storage = {
            rank: SingleProcessFileTensorStorage(data_schema, fpath, mode)
            for rank, fpath in rank_to_fpath.items()
        }
        super().__init__(rank_to_storage)


class MultiProcessRamTensorStorage(MultiProcessTensorStorage):
    def __init__(self, data_schema: Dict[str, SizeData], rank_to_buffer: Dict[int, io.BytesIO]):
        rank_to_storage = {
            rank: SingleProcessRamTensorStorage(data_schema, buf)
            for rank, buf in rank_to_buffer.items()
        }
        super().__init__(rank_to_storage)


def _ram_storage_gather(
    storage: SingleProcessRamTensorStorage, dst_rank: int = 0
) -> MultiProcessRamTensorStorage:
    storage.storage_impl.seek(0, os.SEEK_SET)
    # TODO: overhead, pickling a bytes object, can just pass bytes in a tensor directly
    # see detectron2/utils.comm.py
    data_list = gather(storage.storage_impl.read(), dst=dst_rank)
    if get_rank() != dst_rank:
        return None
    rank_to_buffer = {i: io.BytesIO(data_list[i]) for i in range(len(data_list))}
    storage = MultiProcessRamTensorStorage(storage.data_schema, rank_to_buffer)
    return storage


def _file_storage_gather(
    storage: SingleProcessFileTensorStorage,
    dst_rank: int = 0,
    mode: str = "rb",
) -> MultiProcessFileTensorStorage:
    storage.storage_impl.close()
    fpath_list = gather(storage.fpath, dst=dst_rank)
    if get_rank() != dst_rank:
        return None
    rank_to_fpath = {i: fpath_list[i] for i in range(len(fpath_list))}
    return MultiProcessFileTensorStorage(storage.data_schema, rank_to_fpath, mode)


def storage_gather(
    storage: SingleProcessTensorStorage, dst_rank: int = 0
) -> MultiProcessTensorStorage:
    if isinstance(storage, SingleProcessRamTensorStorage):
        return _ram_storage_gather(storage, dst_rank)
    elif isinstance(storage, SingleProcessFileTensorStorage):
        return _file_storage_gather(storage, dst_rank)
    raise Exception(f"Unsupported storage for gather operation: {storage}")
