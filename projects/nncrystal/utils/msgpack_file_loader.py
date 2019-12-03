from typing import List

import msgpack


def load_msgpack_data(file_path):
    """
    load msgpack from path and return generator returning stored frame data
    """
    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f)

        for unpacked in unpacker:
            yield unpacked


def load_images_entries(file_path, required_index: List, index_list=None):
    """

    :param file_path:
    :param offsets:
    :param index_list: optional prebuilt index
    :return:
    """

    if index_list is None:
        index_list = msgpack_data_index(file_path)

    with open(file_path, "rb") as f:

        for index in required_index:
            offset = index_list[index]
            f.seek(offset)
            unpacker = msgpack.Unpacker(f)
            yield unpacker.unpack()

def msgpack_data_index(file_path):
    """

    :param file_path:
    :return: offset index
    """

    offsets = []
    offset = 0

    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f)
        for _ in unpacker:
            offsets.append(offset)
            offset = unpacker.tell()

    return offsets
