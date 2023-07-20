import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from huicv.json_dataset.coco_ann_utils import GCOCO


def gaussian2d(mu, sigma, M, N, eps=1e-8):
    assert N > 1 and M > 1
    x = np.arange(-0.5, 0.5 + eps, 1. / (M - 1))
    y = np.arange(-0.5, 0.5 + eps, 1. / (N - 1))
    y, x = np.meshgrid(y, x)
    dis = ((x - mu[0]) / sigma[0]) ** 2 + ((y - mu[1]) / sigma[1]) ** 2
    prob = 1.0 / (2 * np.pi * sigma[0] * sigma[1]) * np.exp(- dis / 2)
    return x, y, prob


# mu = (0, 0)
# sigma = (1/4, 1/4)
# M, N = 50, 100
#
# x, y, prob = gaussian2d(mu, sigma, M, N)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(x, y, prob)
# plt.show()


def show_point_distribution(ann_file, img_root):
    def show_first_img_with_ann(ann_file, img_root):
        coco = GCOCO(ann_file)
        imids = list(coco.imgs.keys())
        anns = coco.imgToAnns[imids[0]]
        coco.showAnns(anns, draw_bbox=False, draw_point=True, drwa_image=True, img_root=img_root)
        plt.show()

    jd = json.load(open(ann_file))
    X, Y, ooo,ppp = [], [],[],[]
    num_out = 0
    print(len(jd['annotations']))
    v=0
    for ann in jd['annotations']:
        if 'true_bbox' not in ann:
            print(ann)
        x1, y1, w, h = ann['true_bbox']
        # if w >= 128 or h >= 128:
        #     continue
        if w ==0 or h ==0:
            continue
        if w <= 5 or h <= 5:
            continue
        # x, y = ann['bbox'][0] + 1 / 2 * ann['bbox'][2], ann['bbox'][1] + 1 / 2 * ann['bbox'][3]
        x,y = ann['point']
        if not (x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1):
            # print('out of box')
            num_out += 1
            continue
        # if abs(x-(x1+1/2*w)) >96:
        #     v=v+1
        #     print(v)
        rx, ry = (x - x1) / w, (y - y1) / h
        X.append(rx)
        Y.append(ry)

        ooo.append(abs((w+2*x1)/2-x)/(w/2))
        ppp.append(abs((w + 2 * x1) / 2 - x) / (w / 2))
        # if rx<0:
        #     print(ann)
        if (rx > 10) or ry > 10:
            print(rx, ry, x1, y1, w, h, x, y)
        assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"
    import torch
    print(sum(torch.tensor(X)>0.63))
    print(np.array(X).mean(),np.array(X).std())
    print(np.array(Y).mean(),np.array(Y).std())
    print((np.array(Y)>1).sum(),(np.array(Y)<0).sum())
    plt.hist2d(X, Y, bins=100)
    plt.show()
    print(num_out)

    # show_first_img_with_ann(ann_file, img_root)


# show_point_distribution("data/coco/resize/coarse_annotations/noise_uniform_1//instances_val2017_100x167_coarse.json",
#                         "data/coco/resize/images_100x167_q100")
# show_point_distribution("data/coco/resize/coarse_annotations/noise_rg-0-0-0.25-0.25_1//instances_val2017_100x167_coarse.json",
#                         "data/coco/resize/images_100x167_q100")
# show_point_distribution("data/coco/resize/coarse_annotations/noise_rg-0-0-0.25-0.25_1//instances_train2017_100x167_coarse.json",
#                         "data/coco/resize/images_100x167_q100")
# show_point_distribution("data/coco/resize/coarse_annotations/noise_rg-0-0-0.167-0.167_1//instances_train2017_100x167_coarse.json",
#                         "data/coco/resize/images_100x167_q100")

# show_point_distribution("data/coco/pts_annotation_published/instances_train2014_partial.json",
#                         "data/coco/images")
show_point_distribution("data/coco/coarse_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/instances_train2017_coarse.json",
                        "data/coco/images")
# # show_point_distribution(
#     "../TOV_mmdetection_cache/work_dir/COCO/coarsepointv2/UFO2/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_2ins_r8_8_lr0.01_1x_8b8g/instances_train2017_refine2_2_r8_8.json",
#     "data/coco/images")

# open("data/coco/resize/coarse_annotations/noise_rg-0-0-0.25-0.25_1//instances_train2017_100x167_coarse.json")
