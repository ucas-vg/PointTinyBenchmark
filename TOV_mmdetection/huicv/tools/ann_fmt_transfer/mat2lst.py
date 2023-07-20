import scipy.io as scio
import numpy as np
import os


def _parse_mat(fpath):
    loader = scio.loadmat(fpath)
    #     print(loader['image_info'].shape, loader['image_info'][0, 0].shape,  loader['image_info'][0, 0][0, 0].shape)
    #     print(loader['image_info'][0, 0][0, 0][0].shape)
    #     loader['image_info'][0, 0][0, 0]
    pts = loader['image_info'][0, 0][0, 0][0]
    print(pts)


_parse_mat('/home/hui/dataset/count/shanghai_data/part_A_final/train_data/ground_truth/GT_IMG_1.mat')
