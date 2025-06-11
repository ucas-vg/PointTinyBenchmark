from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch
import tqdm
import numpy as np

def check(coco, res):
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]

                assert ori_ann['id'] == ann['ann_id']
                for key in ['bbox', 'segmentation', 'area', ]:
                    assert key in ori_ann, ori_ann
                    assert key in ann, ann
                    assert ori_ann[key] == ann[key], f"{key}\n\t{ori_ann}\n\t{ann}"


import cv2

from pycocotools import mask as maskUtils
import json


def annToRLE(segm, img_size):
    h, w = img_size
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return rle


def annToMask(segm, img_size):
    if type(segm).__name__ != 'dict':
        rle = annToRLE(segm, img_size)
    else:
        rle = segm
    m = maskUtils.decode(rle)
    return m

def get_bounding_box(mask):
    """
    该函数用于对输入的 mask 矩阵求取最小外接矩形的 bounding box
    :param mask: 输入的二值化 mask 矩阵，形状为 (W, H)
    :return: 包含最小外接矩形信息的 bounding box 列表 [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)  # 找出存在 True 的行
    cols = np.any(mask, axis=0)  # 找出存在 True 的列
    if np.sum(rows) == 0 or np.sum(cols) == 0:
        return [0, 0, 0, 0]  # 若 mask 全为 False，则返回 [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]  # 获取最小和最大的行索引
    x_min, x_max = np.where(cols)[0][[0, -1]]  # 获取最小和最大的列索引
    return [x_min, y_min, x_max, y_max]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_ann", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("det_file", help='such as exp/latest_result.json')
    parser.add_argument("save_ann", help='such as exp/rr_latest_result.json')
    args = parser.parse_args()

    coco = COCO(args.ori_ann)

    res = coco.loadRes(args.det_file)
    iou_sum = 0
    num_sum = 0
    iou_sum_s = 0
    num_sum_s = 0
    iou_sum_m = 0
    num_sum_m = 0
    iou_sum_l = 0
    num_sum_l = 0
    iou_sum_seg = 0
    num_sum_seg = 0
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]
                assert ori_ann['id'] == ann['ann_id'], f"{ori_ann} vs {ann}"

                for key in ['image_id', 'category_id', 'iscrowd']:
                    assert ori_ann[key] == ann[key], key

                # for key in ['bbox', 'segmentation', 'area', ]:
                for key in ['bbox', 'segmentation',  ]:
                    if key == 'segmentation':
                        mask1, mask2 = ori_ann[key], ann[key]
                        m2 = annToMask(mask2, None)
                        m1 = annToMask(mask1, m2.shape)
                        overlap = ((m1 + m2) == 2).sum()
                        union = ((m1 + m2) >= 1).sum()
                        iou_mask = overlap / union
                        iou_sum_seg += iou_mask
                        num_sum_seg += 1

                        ba = torch.tensor(ori_ann['bbox']).unsqueeze(0).float()
                        ba[:, 2:4] = ba[:, 0:2] + ba[:, 2:4]
                        bc = get_bounding_box(m2)
                        bc1 = torch.tensor(bc)[None]
                        iou = bbox_overlaps(ba, bc1)
                        if ori_ann['area'] < 32 * 32:
                            iou_sum_s += iou
                            num_sum_s += 1
                        elif 32 * 32 <= ori_ann['area'] < 64 * 64:
                            iou_sum_m += iou
                            num_sum_m += 1
                        else:
                            iou_sum_l += iou
                            num_sum_l += 1
                        iou_sum += iou
                        num_sum += 1
                        bc=[int(bc[0]),int(bc[1]),int(bc[2]-bc[0]),int(bc[3]-bc[1])]
                        ori_ann['bbox'] = bc
                        ori_ann['area'] = int(m2.sum())
                    # if key == 'bbox':
                    #     #                         print(torch.tensor(ori_ann[key]).unsqueeze(-1).shape)
                    #     ba = torch.tensor(ori_ann[key]).unsqueeze(0).float()
                    #     ba[:, 2:4] = ba[:, 0:2] + ba[:, 2:4]
                    #     bb = torch.tensor(ann[key]).unsqueeze(0).float()
                    #     bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
                    #     iou = bbox_overlaps(ba, bb)
                    #     if ori_ann['area'] < 32 * 32:
                    #         iou_sum_s += iou
                    #         num_sum_s += 1
                    #     elif 32 * 32 <= ori_ann['area'] < 64 * 64:
                    #         iou_sum_m += iou
                    #         num_sum_m += 1
                    #     else:
                    #         iou_sum_l += iou
                    #         num_sum_l += 1
                    #     iou_sum += iou
                    #     num_sum += 1

                        ori_ann[key] = ann[key]
                ## add by fei
                ori_ann['ann_weight'] = ann['score']
    mean_iou = iou_sum / num_sum
    print('detection:', mean_iou, num_sum, iou_sum_s / num_sum_s, iou_sum_m / num_sum_m, iou_sum_l / num_sum_l,
          num_sum_s, num_sum_m,
          num_sum_l)
    # print('segmentation:', iou_sum_seg / num_sum_seg)
    # f = open(args.save_ann.split('.')[0] + '.txt', 'w')
    # f.writelines(mean_iou+' '+num_sum+' '+iou_sum_s+' '+num_sum_s+' '+iou_sum_m/num_sum_m+' '+iou_sum_l/num_sum_l+' '+num_sum_s+' '+num_sum_m+' '+num_sum_l)
    # check(coco, res)
    json.dump(coco.dataset, open(args.save_ann, 'w'))
