import json
from collections import defaultdict
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
mode = 2

# 根目录文件
root = 'data/coco/images/'

gt_path = "../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json"
# result_path = "coco/V_16_coco17_quasi_center_point/inference/my_coco_2014_minival/bbox.json"

# with open(result_path,"r") as f:
#     result = json.load(f)

with open(gt_path, "r") as f:
    gt = json.load(f)

# 展现的100个图片的列表
show_img = []
for x in gt['images'][:100]:
    show_img.append(x['file_name'])

# 这个函数完全没必要写，因为这里的对应关系很简单
# 把id填充0至12位即可，不需要写循环对应
img2id = defaultdict()
for x in gt['images']:
    if x['file_name'] in show_img:
        img2id[x['file_name']] = x['id']

# 把要展示图片的框位置和标注记录下来
gt_bbox = defaultdict(list)
gt_class = defaultdict(list)
for x in gt['annotations']:
    if x['image_id'] in img2id.values():
        gt_bbox[x['image_id']].append(x['bbox'])
        gt_class[x['image_id']].append(coco_id_name_map[x['category_id']])

# #自己检测结果的可视化的记录数据
# id2bbox = defaultdict(list)
# id2class = defaultdict(list)
# id2score = defaultdict(list)
# score_thr=0.6
# for x in result:
#     if x['image_id'] in img2id.values() and x['score'] > score_thr:
#         id2bbox[x['image_id']].append(x['bbox'])
#         id2class[x['image_id']].append(coco_id_name_map[x['category_id']])
#         id2score[x['image_id']].append(x['score'])


# 展示图片，保存
# 定义一个展示gt或result的变量
gt_or_result = 1

for img_name in show_img[:100]:
    img_id = img2id[img_name]
    if gt_or_result == 1:
        bbox = gt_bbox[img_id]
        classes = gt_class[img_id]
    else:
        bbox = id2bbox[img_id]
        classes = id2class[img_id]
        scores = id2score[img_id]

    plt.figure(dpi=250)
    ax = plt.gca()
    ax.axis('off')

    img = Image.open(os.path.join(root, img_name))
    img = np.array(img)

    color = (255, 140, 0)
    color_1 = (1, 140 / 255, 0)
    name_out = 'p2bnet'
    # 按照gt和result给予不同的标注
    if gt_or_result == 1:
        for i, bb in enumerate(bbox):
            bb = [int(x) for x in bb]
            top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), color,
                                4)  # blue(0,229,238) yellow(255,215,0) green(127,255,0) orange(255,140,0)

            label_text = classes[i]

            ax.text(
                bb[0],
                bb[1],
                f'{label_text}',
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color=color_1,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left')

    else:
        for i, bb in enumerate(bbox):
            bb = [int(x) for x in bb]
            top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (127, 255, 0), 4)  # (0,229,238)
            cla = classes[i]
            sco = scores[i]
            label_text = cla + '|' + f'{sco:.02f}'
            ax.text(
                bb[0],
                bb[1],
                f'{label_text}',
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color=(1, 0, 0),
                fontsize=13,
                verticalalignment='top',
                horizontalalignment='left')
    # 保存图片
    plt.imshow(img)
    if not os.path.exists('vis_pt_bbox_' + name_out):
        os.makedirs('vis_pt_bbox_' + name_out)
    if gt_or_result == 1:
        plt.savefig(os.path.join('vis_pt_bbox_' + name_out, img_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join('vis_pt_bbox_' + name_out, img_name), bbox_inches='tight')
