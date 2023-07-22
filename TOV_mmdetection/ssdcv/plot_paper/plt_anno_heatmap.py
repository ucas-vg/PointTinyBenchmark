from pycocotools.coco import COCO
from .plt_paper_config import *

#  PYTHONPATH=.:$PYTHONPATH  python exp/log_utils/plt_anno_heatmap.py

"""
3D heat map
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import warnings


class MyNorm(matplotlib.colors.Normalize):
    def __init__(self, value_to_red=1):
        self.value_to_red = value_to_red

    def __call__(self, mask_array):
        if mask_array.max() > self.value_to_red:
            warnings.warn(f"exist value {mask_array.max()} > {self.value_to_red} in matplotlib plot")
        mask_array /= self.value_to_red
        return mask_array


def plot_3d_heatmap(data, bins=(100, 100), zbound=(0, 1000)):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=FIGSIZE)

    data_x, data_y = data[:, 0], data[:, 1]
    C, X, Y, _ = plt.hist2d(data_x, data_y, bins=bins)
    plt.clf()
    X = (X[:-1] + X[1:]) / 2
    Y = (Y[:-1] + Y[1:]) / 2

    X, Y = np.meshgrid(X, Y)

    ax = Axes3D(fig)
    ax.plot_surface(X, Y, C, cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
    #     ax.set_xlabel("x")
    ax.axis('off')

    ax.set_zbound(*zbound)

    ax.set_xlim()
    plt.title("3D", fontsize=TITLE_FONTSIZE)


def plt_2d_heatmap(data, bins=(100, 100), *args, **kwargs):
    positions = data
    plt.hist2d(positions[:, 0], positions[:, 1], bins=(100, 100))
    plt.axes().set_xlabel("var: ({},{}), mean: ({}, {})".format(
        round(np.std(positions[:, 0]) ** 2, 3), round(np.std(positions[:, 1]) ** 2, 3),
        round(positions[:, 0].mean(), 2), round(positions[:, 1].mean(), 2)
    ))
    plt.axes().yaxis.set_ticks_position('left')  # 将x轴的位置设置在顶部
    plt.axes().invert_yaxis()


if __name__ == "__main__":
    # get_json_file({"noise": "uniform", "corner": (640, 640, 100, 100), "method": "pseuw32h32", "round": 1})
    # get_dataset_name_path([
    #     # # noise visDrone Person
    #     {"noise": "uniform", "corner": (640, 640, 100, 100), "method": "pseuw32h32", "round": [None, 1, 2, 3]},
    #     {"noise": "rg0.0_0.250", "corner": (640, 640, 100, 100), "method": "pseuw32h32", "round": [None, 1, 2, 3]},
    # ])

    from exp.log_utils.dataset_utils import get_json_file, get_point_positions, deal_invalid_position, size_filter
    from exp.log_utils.dataset_utils import get_dataset_name_path
    import json
    import matplotlib.pyplot as plt

    size_th = (0, 10000)

    for r in [None, 1, 2, 3]:
        anno_data = COCO(
            get_json_file({"noise": "uniform", "corner": (640, 640, 100, 100), "method": "pseuw32h32", "round": r}))
        print(anno_data.anns[0])
        positions = get_point_positions(anno_data)
        positions = deal_invalid_position(positions)
        positions = size_filter(positions, size_th)

        # plt_2d_heatmap(positions, bins=(100, 100))
        plot_3d_heatmap(positions, bins=(100, 100))
