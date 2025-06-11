from huicv.json_dataset.coco_ann_utils import GCOCO
import json
import numpy as np
import cv2


def draw_occlusion(coco, occ_rate, img_fold, img_out_fold):
    imids = list(coco.imgs.keys())
    n=0
    for id in imids:
        n+=1
        anns = coco.imgToAnns[id]
        num_gt = len(anns)
        pts = [ann['point'] for ann in anns]
        file_name = coco.imgs[id]['file_name']
        img = cv2.imread(img_fold + file_name)
        for ann in anns:
            true_bbox = ann['true_bbox']
            occ_bbox = ann['occ_bbox']
            point = ann['point']
            if true_bbox[2] < 1 or true_bbox[3] < 1:
                pass
            else:
                img = cv2.rectangle(img, (round(occ_bbox[0]), round(occ_bbox[1])),
                                    (round(occ_bbox[0] + occ_bbox[2]), round(occ_bbox[1] + occ_bbox[3])),
                                    color=(0, 0, 0), thickness=-1)
                # img = cv2.circle(img, (round(point[0]), round(point[1])), 3, (0, 255, 255), thickness=-1)
                # img = cv2.rectangle(img, (round(true_bbox[0]), round(true_bbox[1])),
                #                     (round(true_bbox[0] + true_bbox[2]), round(true_bbox[1] + true_bbox[3])),
                #                     color=(0, 0, 255), thickness=3)
        # cv2.namedWindow('img')
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        print(n)
        cv2.imwrite(img_out_fold + file_name,img)


ann_file = 'data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/occlusion/occlusion_instances_train2017_coarse_0.3.json'
img_fold = 'data/coco/images/'
img_out_fold = 'data/coco/occlusion/images/'
coco = GCOCO(ann_file)
occ_rate = 0.3  ## setting as ann_file
draw_occlusion(coco, occ_rate, img_fold, img_out_fold)
