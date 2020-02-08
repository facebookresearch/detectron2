import matplotlib.pyplot as plt
import numpy as np

from typing import Union, List


class HistorgramVisualizer:

    @staticmethod
    def show(x: Union[np.ndarray, List],
             y: Union[np.ndarray, List],
             xlabel: str = "",
             ylabel: str = "",
             title: str = "",
             xlim: Union[None, List] = None,
             ylim: Union[None, List] = None,
             is_save: bool = False):

        plt.figure()
        print(x.shape)
        print(y.shape)
        print(x)
        print(y)
        plt.bar(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)

        if is_save:
            plt.savefig("{}.png".format(title))

        plt.show()
