import io

import numpy as np
from matplotlib import pyplot as plt
from utils import msgpack_data_index, load_images_entries
from PIL import Image


def montage_previewer(file_path, num_row: int, num_col: int, skip_first_n: int = None, skip_last_n: int = None):
    index = msgpack_data_index(file_path)

    length = len(index)
    num_images = num_row * num_col

    interval = int(length / num_images)

    start = 0 if skip_first_n is None else skip_first_n
    end = length if skip_last_n is None else length - skip_last_n

    assert end > start and end - start > num_images

    image_index = np.arange(start, end, interval).tolist()

    image_entries = load_images_entries(file_path, image_index, index)

    fig: plt.Figure
    fig, ax = plt.subplots(nrows=num_row, ncols=num_col)

    for row in ax:
        col: plt.Axes
        for col in row:
            im = next(image_entries)

            im: Image.Image = Image.open(io.BytesIO(im[b"data"]))

            col.imshow(im, cmap="gray")
            col.axis("off")
    fig.subplots_adjust(wspace=0, hspace=0)

    return fig
