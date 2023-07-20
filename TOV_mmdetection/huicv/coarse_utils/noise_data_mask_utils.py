import math

import numpy as np
import os.path as osp
import json
from copy import deepcopy
import os
from huicv.interactivate.path_utils import makedirs_if_not_exist
from huicv.interactivate.cmd import input_yes
from huicv.json_dataset.coco_ann_utils import dump_coco_annotation
import matplotlib.pyplot as plt

from pycocotools.coco import maskUtils
from huicv.json_dataset.coco_ann_utils import GCOCO
from tqdm import tqdm
from math import ceil, floor


def get_object_mask(ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    # t = self.imgs[ann['image_id']]
    # h, w = t['height'], t['width']
    bbox = ann['bbox']
    x1, y1, w, h = bbox
    ann = GCOCO.translate_ann(ann, -x1, -y1)
    segm = ann['segmentation']

    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']

    m = maskUtils.decode(rle)
    return m


def choose_one(alist):
    i = int(np.random.uniform(0, 1) * len(alist))
    if i == len(alist):
        i -= 1
    return alist[i]


def clip(x, minx, maxx):
    if x < minx:
        return minx
    elif x > maxx:
        return maxx
    else:
        return x


def sample_by_distribution(dist_hist):
    dist_cumsum = np.cumsum(dist_hist)
    r = np.random.uniform(0, 1)
    for i, p in enumerate(dist_cumsum):
        if r < p:
            return i
    return len(dist_cumsum) - 1


def gaussian2d(mu, sigma, W, H, eps=1e-8):
    assert W > 1 and H > 1
    x = np.arange(-0.5, 0.5 + eps, 1. / (W - 1))
    y = np.arange(-0.5, 0.5 + eps, 1. / (H - 1))
    x, y = np.meshgrid(x, y)
    dis = ((x - mu[0]) / sigma[0]) ** 2 + ((y - mu[1]) / sigma[1]) ** 2
    prob = 1.0 / (2 * np.pi * sigma[0] * sigma[1]) * np.exp(- dis / 2)
    return x, y, prob


def random_choose_in_seg_points(ann):
    seg = ann['segmentation']
    if isinstance(seg, list):
        counts = seg[0]
        n = len(counts) // 2 * 2
        pts = np.array(counts[:n]).reshape(-1, 2)
        return choose_one(pts)
    elif isinstance(seg['counts'], list):  # choose center
        x1, y1, w, h = ann['bbox']
        return x1 + w / 2, y1 + h / 2
    else:
        raise ValueError("")


def random_choose_in_mask(ann, size_range, rand_type, **rand_kwargs):
    x1, y1, w, h = ann['bbox']
    if h < 2 or w < 2:
        return random_choose_in_seg_points(ann)
    else:
        mask = get_object_mask(ann)
    if mask.sum() == 0:
        return random_choose_in_seg_points(ann)
    if rand_type == 'uniform':
        idxs = np.stack(np.nonzero(mask), axis=1)
        dh, dw = choose_one(idxs)
        x, y = x1 + dw, y1 + dh
    elif rand_type == 'range_gaussian' or rand_type == 'center_gaussian':
        mu, sigma = rand_kwargs['mu'], rand_kwargs['sigma']
        h, w = mask.shape
        if h < 2 or w < 2:
            return random_choose_in_seg_points(ann)
        _, _, prob = gaussian2d(mu, sigma, w, h)
        prob = prob * mask
        if rand_type == 'center_gaussian':
            center = np.zeros(prob.shape)
            h0 = math.floor((h - h * size_range) / 2)
            w0 = math.floor((w - w * size_range) / 2)
            import cv2
            center = cv2.ellipse(center, (round(w / 2), round(h / 2)),
                                 (round(min(96, w * size_range / 2)), round(min(96, h * size_range / 2))), 0,
                                 0, 360, (1.0),
                                 -1)
            # center[h0: -h0, w0: -w0] = 1.0
            prob = prob if (prob * center).sum() == 0 else prob * center
        if prob.sum() == 0:
            return random_choose_in_seg_points(ann)
        prob /= prob.sum()

        idx = sample_by_distribution(prob.reshape(-1, ))
        dh, dw = idx // prob.shape[1], idx % prob.shape[1]
        x, y = x1 + dw, y1 + dh
    else:
        raise NotImplementedError

    x1, y1, w, h = ann['bbox']
    assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"
    x, y = clip(x, x1, x1 + w), clip(y, y1, y1 + h)
    return x, y


def add_rand_points_in_mask(jd, size_range=1.0, rand_type='uniform', rand_kwargs={}):
    if isinstance(size_range, str):
        size_range = eval(size_range)
    assert 0 < size_range <= 1.0

    anns = jd['annotations']
    offset = np.random.uniform(-0.5, 0.5, size=(len(anns), 2))
    for i, ann in tqdm(enumerate(anns)):
        x, y = random_choose_in_mask(ann, size_range, rand_type, **rand_kwargs)
        x1, y1, w, h = ann['bbox']
        assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"

        x, y = x + offset[i, 0], y + offset[i, 1]  # shake in neighbour
        ann['point'] = [x, y]

        x1, y1, w, h = ann['bbox']
        assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"


def dump_json(jd, ann_file, success_info=None):
    if osp.exists(ann_file):
        if not input_yes(f"{ann_file} have exists, do you want to overwrite? (y/n)"):
            return False
    makedirs_if_not_exist(ann_file)
    dump_coco_annotation(jd, ann_file)
    if success_info is not None:
        print(success_info)
    return True


def replace_key_to_in_ann(jd, old_key, new_key):
    for ann in jd['annotations']:
        ann[new_key] = ann[old_key]
        del ann[old_key]


def generate_noisept_dataset(ann_file, save_ann=None, *args, **kwargs):
    import json
    jd = json.load(open(ann_file))
    add_rand_points_in_mask(jd, *args, **kwargs)
    replace_key_to_in_ann(jd, 'bbox', 'true_bbox')
    dump_json(jd, save_ann, 'save noisept annotation file in {}'.format(save_ann))


def show_point_distribution(ann_file):
    jd = json.load(open(ann_file))
    X, Y = [], []
    for ann in jd['annotations']:
        x1, y1, w, h = ann['true_bbox']
        x, y = ann['point']
        rx, ry = (x - x1) / w, (y - y1) / h
        X.append(rx)
        Y.append(ry)
        if (rx > 10) or ry > 10:
            print(rx, ry, x1, y1, w, h, x, y)
        assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"

    plt.hist2d(X, Y)
    plt.show()


def demo_test():
    from PIL import Image

    img_root = "data/coco/resize/images_100x167_q100"
    ann_file = "data/coco/resize/annotations/instances_val2017_100x167.json"
    # ann_file = "data/coco/annotations/instances_val2017.json"

    coco = GCOCO(ann_file)

    imids = list(coco.imgs.keys())
    anns = coco.imgToAnns[imids[0]]
    # im_info = coco.imgs[imids[0]]
    # im_path = "{}/{}".format(img_root, im_info['file_name'])
    # img = Image.open(im_path)
    # img = np.array(img)
    # plt.imshow(img)

    # print(coco.anns[900100374545])
    # print(coco.imgs[coco.anns[900100374545]['image_id']])

    print(anns)
    add_rand_points_in_mask({"annotations": anns})
    coco.showAnns(anns, draw_bbox=False, draw_point=True, drwa_image=True, img_root=img_root)
    plt.show()


if __name__ == '__main__':
    # demo_test()

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('task', help='task, can be "generate_noisept_dataset"', default="generate_noisept_dataset")
    parser.add_argument('ann', help='ann_file', default="data/coco/annotations/instances_train2017.json")
    parser.add_argument('save_ann', help='save_ann', default="test.json")
    parser.add_argument('--size_range', help='size_range, only for generate_noisept_dataset, 1.0 means in bounding box',
                        default=0.25, type=float)
    parser.add_argument('--rand_type', help='random method, only for generate_noisept_dataset',
                        default='center_gaussian')
    parser.add_argument('--range_gaussian_mu', help='range_gaussian_mu only used while rand_type is range_gaussian',
                        default=0, type=str)
    parser.add_argument('--range_gaussian_sigma',
                        help='range_gaussian_sigma only used while rand_type is range_gaussian',
                        default=1, type=str)
    parser.add_argument('--show', action='store_true', help='whether show distribution of points.')
    args = parser.parse_args()

    mu, sigma = args.range_gaussian_mu, args.range_gaussian_sigma
    if isinstance(mu, str):
        mu = eval(mu)
    if isinstance(sigma, str):
        sigma = eval(sigma)
    if isinstance(mu, (int, float)):
        mu = (mu, mu)
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    if args.task == 'generate_noisept_dataset':
        # print(args)
        assert len(args.save_ann) > 0
        # python mmdet/datasets/noise_data_utils.py
        # dataset = GCOCO(args.ann)
        generate_noisept_dataset(
            args.ann,
            args.save_ann,
            size_range=args.size_range,
            rand_type=args.rand_type,
            rand_kwargs=dict(mu=mu, sigma=sigma)
        )
        if args.show:
            show_point_distribution(args.save_ann)
    else:
        raise ValueError
