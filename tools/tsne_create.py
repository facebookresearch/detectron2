import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

color_map = ['r','y','k','g','b','m','c'] # 7个类，准备7种颜色
def plot_embedding_2D(data, label, title):
    """

    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig



import random
from collections import defaultdict
if __name__ == "__main__":
    labels = torch.load("tsne_labels313.pth").cpu()
    features = torch.load("tsne_feauture313.pth").cpu()

    result = defaultdict(list)

    for i in range(6):
        area = labels==i
        cls_feature = features[area]

        B, D = cls_feature.shape
        Num = min(500, B)
        index = torch.LongTensor(random.sample(range(B), Num))

        cls_feature = torch.index_select(cls_feature, 0, index)

        label = torch.full([Num,], i)

        result["labels"].append(label)
        result['features'].append(cls_feature)

    tsne = TSNE(n_components=2, verbose=1)

    result_l = torch.cat(result["labels"], dim=0)
    result_f = torch.cat(result['features'], dim=0)
    # labels = torch.cat(labels, dim=0)
    # features = torch.cat(features, dim=0)
    # features = features[:10000,:]
    # labels = labels[:10000,]


    result_2D = tsne.fit_transform(result_f)
    fig1 = plot_embedding_2D(result_2D, result_l, 't-SNE')
    plt.savefig("tsne1.png")