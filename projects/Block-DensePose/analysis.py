import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_file = 'output/inference/confidence.npy'
    data = np.load(data_file)

    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    fig = plt.figure()

    plt.xlabel('confidence')
    plt.ylabel('distance')
    plt.scatter(data[:, 1], data[:, 0], c='#638DB7', marker='.', s=10)
    plt.legend('baseline')
    plt.show()
