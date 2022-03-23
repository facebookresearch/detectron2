# Copyright (c) Facebook, Inc. and its affiliates.

import io
import tempfile
import unittest
from contextlib import ExitStack
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm

from densepose.evaluation.tensor_storage import (
    SingleProcessFileTensorStorage,
    SingleProcessRamTensorStorage,
    SizeData,
    storage_gather,
)


class TestSingleProcessRamTensorStorage(unittest.TestCase):
    def test_read_write_1(self):
        schema = {
            "tf": SizeData(dtype="float32", shape=(112, 112)),
            "ti": SizeData(dtype="int32", shape=(4, 64, 64)),
        }
        # generate data which corresponds to the schema
        data_elts = []
        torch.manual_seed(23)
        for _i in range(3):
            data_elt = {
                "tf": torch.rand((112, 112), dtype=torch.float32),
                "ti": (torch.rand(4, 64, 64) * 1000).to(dtype=torch.int32),
            }
            data_elts.append(data_elt)
        storage = SingleProcessRamTensorStorage(schema, io.BytesIO())
        # write data to the storage
        for i in range(3):
            record_id = storage.put(data_elts[i])
            self.assertEqual(record_id, i)
        # read data from the storage
        for i in range(3):
            record = storage.get(i)
            self.assertEqual(len(record), len(schema))
            for field_name in schema:
                self.assertTrue(field_name in record)
                self.assertEqual(data_elts[i][field_name].shape, record[field_name].shape)
                self.assertEqual(data_elts[i][field_name].dtype, record[field_name].dtype)
                self.assertTrue(torch.allclose(data_elts[i][field_name], record[field_name]))


class TestSingleProcessFileTensorStorage(unittest.TestCase):
    def test_read_write_1(self):
        schema = {
            "tf": SizeData(dtype="float32", shape=(112, 112)),
            "ti": SizeData(dtype="int32", shape=(4, 64, 64)),
        }
        # generate data which corresponds to the schema
        data_elts = []
        torch.manual_seed(23)
        for _i in range(3):
            data_elt = {
                "tf": torch.rand((112, 112), dtype=torch.float32),
                "ti": (torch.rand(4, 64, 64) * 1000).to(dtype=torch.int32),
            }
            data_elts.append(data_elt)
        # WARNING: opens the file several times! may not work on all platforms
        with tempfile.NamedTemporaryFile() as hFile:
            storage = SingleProcessFileTensorStorage(schema, hFile.name, "wb")
            # write data to the storage
            for i in range(3):
                record_id = storage.put(data_elts[i])
                self.assertEqual(record_id, i)
            hFile.seek(0)
            storage = SingleProcessFileTensorStorage(schema, hFile.name, "rb")
            # read data from the storage
            for i in range(3):
                record = storage.get(i)
                self.assertEqual(len(record), len(schema))
                for field_name in schema:
                    self.assertTrue(field_name in record)
                    self.assertEqual(data_elts[i][field_name].shape, record[field_name].shape)
                    self.assertEqual(data_elts[i][field_name].dtype, record[field_name].dtype)
                    self.assertTrue(torch.allclose(data_elts[i][field_name], record[field_name]))


def _find_free_port():
    """
    Copied from detectron2/engine/launch.py
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(main_func, nprocs, args=()):
    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    # dist_url = "env://"
    mp.spawn(
        distributed_worker, nprocs=nprocs, args=(main_func, nprocs, dist_url, args), daemon=False
    )


def distributed_worker(local_rank, main_func, nprocs, dist_url, args):
    dist.init_process_group(
        backend="gloo", init_method=dist_url, world_size=nprocs, rank=local_rank
    )
    comm.synchronize()
    assert comm._LOCAL_PROCESS_GROUP is None
    pg = dist.new_group(list(range(nprocs)))
    comm._LOCAL_PROCESS_GROUP = pg
    main_func(*args)


def ram_read_write_worker():
    schema = {
        "tf": SizeData(dtype="float32", shape=(112, 112)),
        "ti": SizeData(dtype="int32", shape=(4, 64, 64)),
    }
    storage = SingleProcessRamTensorStorage(schema, io.BytesIO())
    world_size = comm.get_world_size()
    rank = comm.get_rank()
    data_elts = []
    # prepare different number of tensors in different processes
    for i in range(rank + 1):
        data_elt = {
            "tf": torch.ones((112, 112), dtype=torch.float32) * (rank + i * world_size),
            "ti": torch.ones((4, 64, 64), dtype=torch.int32) * (rank + i * world_size),
        }
        data_elts.append(data_elt)
    # write data to the single process storage
    for i in range(rank + 1):
        record_id = storage.put(data_elts[i])
        assert record_id == i, f"Process {rank}: record ID {record_id}, expected {i}"
    comm.synchronize()
    # gather all data in process rank 0
    multi_storage = storage_gather(storage)
    if rank != 0:
        return
    # read and check data from the multiprocess storage
    for j in range(world_size):
        for i in range(j):
            record = multi_storage.get(j, i)
            record_gt = {
                "tf": torch.ones((112, 112), dtype=torch.float32) * (j + i * world_size),
                "ti": torch.ones((4, 64, 64), dtype=torch.int32) * (j + i * world_size),
            }
            assert len(record) == len(schema), (
                f"Process {rank}: multi storage record, rank {j}, id {i}: "
                f"expected {len(schema)} fields in the record, got {len(record)}"
            )
            for field_name in schema:
                assert field_name in record, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name} not in the record"
                )

                assert record_gt[field_name].shape == record[field_name].shape, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, expected shape {record_gt[field_name].shape} "
                    f"got {record[field_name].shape}"
                )
                assert record_gt[field_name].dtype == record[field_name].dtype, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, expected dtype {record_gt[field_name].dtype} "
                    f"got {record[field_name].dtype}"
                )
                assert torch.allclose(record_gt[field_name], record[field_name]), (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, tensors are not close enough:"
                    f"L-inf {(record_gt[field_name]-record[field_name]).abs_().max()} "
                    f"L1 {(record_gt[field_name]-record[field_name]).abs_().sum()} "
                )


def file_read_write_worker(rank_to_fpath):
    schema = {
        "tf": SizeData(dtype="float32", shape=(112, 112)),
        "ti": SizeData(dtype="int32", shape=(4, 64, 64)),
    }
    world_size = comm.get_world_size()
    rank = comm.get_rank()
    storage = SingleProcessFileTensorStorage(schema, rank_to_fpath[rank], "wb")
    data_elts = []
    # prepare different number of tensors in different processes
    for i in range(rank + 1):
        data_elt = {
            "tf": torch.ones((112, 112), dtype=torch.float32) * (rank + i * world_size),
            "ti": torch.ones((4, 64, 64), dtype=torch.int32) * (rank + i * world_size),
        }
        data_elts.append(data_elt)
    # write data to the single process storage
    for i in range(rank + 1):
        record_id = storage.put(data_elts[i])
        assert record_id == i, f"Process {rank}: record ID {record_id}, expected {i}"
    comm.synchronize()
    # gather all data in process rank 0
    multi_storage = storage_gather(storage)
    if rank != 0:
        return
    # read and check data from the multiprocess storage
    for j in range(world_size):
        for i in range(j):
            record = multi_storage.get(j, i)
            record_gt = {
                "tf": torch.ones((112, 112), dtype=torch.float32) * (j + i * world_size),
                "ti": torch.ones((4, 64, 64), dtype=torch.int32) * (j + i * world_size),
            }
            assert len(record) == len(schema), (
                f"Process {rank}: multi storage record, rank {j}, id {i}: "
                f"expected {len(schema)} fields in the record, got {len(record)}"
            )
            for field_name in schema:
                assert field_name in record, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name} not in the record"
                )

                assert record_gt[field_name].shape == record[field_name].shape, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, expected shape {record_gt[field_name].shape} "
                    f"got {record[field_name].shape}"
                )
                assert record_gt[field_name].dtype == record[field_name].dtype, (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, expected dtype {record_gt[field_name].dtype} "
                    f"got {record[field_name].dtype}"
                )
                assert torch.allclose(record_gt[field_name], record[field_name]), (
                    f"Process {rank}: multi storage record, rank {j}, id {i}: "
                    f"field {field_name}, tensors are not close enough:"
                    f"L-inf {(record_gt[field_name]-record[field_name]).abs_().max()} "
                    f"L1 {(record_gt[field_name]-record[field_name]).abs_().sum()} "
                )


class TestMultiProcessRamTensorStorage(unittest.TestCase):
    def test_read_write_1(self):
        launch(ram_read_write_worker, 8)


class TestMultiProcessFileTensorStorage(unittest.TestCase):
    def test_read_write_1(self):
        with ExitStack() as stack:
            # WARNING: opens the files several times! may not work on all platforms
            rank_to_fpath = {
                i: stack.enter_context(tempfile.NamedTemporaryFile()).name for i in range(8)
            }
            launch(file_read_write_worker, 8, (rank_to_fpath,))
