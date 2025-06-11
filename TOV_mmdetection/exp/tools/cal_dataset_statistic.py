import json
import numpy as np
from collections import defaultdict
from huicv.coarse_utils.noise_data_utils import mean_ann_size_of_class
from huicv.json_dataset.coco_ann_utils import GCOCO
# from tqdm import tqdm


def s(w, h):
    mine = min(w, h)
    maxe = max(w, h)
    return min(1333 / maxe, 800/mine)


def mean_ann_size(json_file):
    jd = json.load(open(json_file))
    images = {im_info['id']: im_info for im_info in jd['images']}
    scale_factor = {im: s(info["width"], info["height"]) for im, info in images.items()}

    sizes = []
    for ann in jd['annotations']:
        x, y, w, h = ann['bbox']
        sizes.append(((w * h)**0.5))
    sizes = [s for s in sizes if s >= 2]
    print(np.mean(sizes), np.exp(np.mean(np.log(sizes))))


# def mean_ann_size_of_class(json_file):
#     jd = json.load(open(json_file))
#     cls_sizes = defaultdict(list)
#
#     for ann in jd['annotations']:
#         x, y, w, h = ann['bbox']
#         if w >= 2 and h >=2:
#             cls_sizes[ann['category_id']].append([w, h])
#
#     cls_mean_size = {}
#     for cls, sizes in cls_sizes.items():
#         sizes = np.array(sizes)
#         sizes = (sizes[:, 0] * sizes[:, 1]) ** 0.5
#         print(cls, np.mean(sizes), np.exp(np.mean(np.log(sizes))))
#     # print(np.mean(sizes), np.exp(np.mean(np.log(sizes))))


# mean_ann_size("data/coco/annotations/instances_train2017.json")

gcoco = GCOCO("data/coco/resize/annotations/instances_train2017_100x167.json")
cls_mean_size = mean_ann_size_of_class(gcoco.dataset, bbox_key='bbox', use='wh')
label_mean_size = [None] * len(cls_mean_size)
for cls, mean_size in cls_mean_size.items():
    label_mean_size[gcoco.cat2label[cls]] = mean_size
json.dump({"label2mean_size": label_mean_size}, open("exp/tinycocotrain_label_mean_wh.json", 'w'))
