from huicv.json_dataset.coco_ann_utils import GCOCO
import json
import numpy as np


def x1y1x2y2_2_x1y1wh(bbox):
    # bbox = [np.float16(i) for i in bbox]
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def x1y1wh_2_x1y1x2y2(bbox):
    # bbox = [np.float16(i) for i in bbox】
    # return [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def occlusion_single(true_bbox, occ_rate):
    x1, y1, w, h = true_bbox
    x2, y2 = x1 + w, y1 + h
    area = w * h
    occlusion_area = area * occ_rate
    a = np.random.rand(3)
    x_min = x1 + occ_rate / 2 * w * 1.2
    y_min = y1 + occ_rate / 2 * h * 1.2
    x_max = x2 - occ_rate / 2 * w * 1.2
    y_max = y2 - occ_rate / 2 * h * 1.2
    x_c_occ = x_min + a[0] * (x_max - x_min)
    y_c_occ = y_min + a[1] * (y_max - y_min)

    max_w = min(2 * (x2 - x_c_occ), 2 * (x_c_occ - x1))
    max_h = min(2 * (y2 - y_c_occ), 2 * (y_c_occ - y1))
    max_w = 1 if max_w < 1 else max_w
    max_h = 1 if max_h < 1 else max_h

    # from warnings import simplefilter
    # simplefilter('error')
    min_w = occlusion_area / max_h
    w_occ = min_w + (max_w - min_w) * a[2]

    h_occ = occlusion_area / w_occ
    x_1_occ, y_1_occ, x_2_occ, y_2_occ = x_c_occ - w_occ / 2, y_c_occ - h_occ / 2, x_c_occ + w_occ / 2, y_c_occ + h_occ / 2
    return [x_1_occ, y_1_occ, w_occ, h_occ]


def occlusion_single_v2(true_bbox, occ_rate, point):
    true_bbox = np.array(true_bbox)
    x1, y1, w, h = true_bbox
    x2, y2 = x1 + w, y1 + h
    area = w * h

    occlusion_area = area * occ_rate

    # xa, xb, corx = x1, point[0], 0 if point[0] - x1 > x2 - point[0] else point[0], x2, 1
    # ya, yb, cory = y1, point[1], 0 if point[1] - y1 > y2 - point[1] else point[1], y2, 1
    a = np.random.rand(1)
    if point[0] - x1 >= x2 - point[0] and point[1] - y1 >= y2 - point[1]:
        x_occ = a * (point[0] - x1) + x1
        y_occ = y1 + occlusion_area / (x_occ - x1)
        return np.array([x1, y1, x_occ, y_occ])
    elif point[0] - x1 >= x2 - point[0] and point[1] - y1 < y2 - point[1]:
        x_occ = a * (point[0] - x1) + x1
        y_occ = y2 - occlusion_area / (x_occ - x1)
        return np.array([x1, y_occ, x_occ, y2])
    elif point[0] - x1 < x2 - point[0] and point[1] - y1 >= y2 - point[1]:
        x_occ = a * (x2 - point[0]) + point[0]
        y_occ = y1 + occlusion_area / (x2 - x_occ)
        return np.array([x_occ, y1, x2, y_occ])
    elif point[0] - x1 < x2 - point[0] and point[1] - y1 < y2 - point[1]:
        x_occ = a * (x2 - point[0]) + point[0]
        y_occ = y2 - occlusion_area / (x2 - x_occ)
        return np.array([x_occ, y_occ, x2, y2])


def pts_in_occlusion(occ_bbox, pts):
    occ_bbox = x1y1wh_2_x1y1x2y2(occ_bbox)
    for pt in pts:
        if occ_bbox[0] < pt[0] < occ_bbox[2] and occ_bbox[1] < pt[1] < occ_bbox[3]:
            print(pt)
            return True
    return False


def occlusion(coco, occ_rate):
    X, Y = [], []
    # anns = coco.anns
    # ann = coco.imgs

    n = 0
    imids = list(coco.imgs.keys())
    for id in imids:
        anns = coco.imgToAnns[id]
        num_gt = len(anns)
        pts = [ann['point'] for ann in anns]
        for ann in anns:
            true_bbox = ann['true_bbox']
            if true_bbox[2] < 1 or true_bbox[3] < 1:
                ann['occ_bbox'] = true_bbox
            else:
                point = ann['point']
                x1, y1, w, h = true_bbox
                x2, y2 = x1 + w, y1 + h
                # occ_bbox = occlusion_single(true_bbox, occ_rate)
                occ_bbox = occlusion_single(true_bbox, occ_rate)

                # while occ_bbox[0] < x1 - 1 or occ_bbox[1] < y1 - 1 \
                #         or occ_bbox[2] > x2 + 1 or occ_bbox[3] > y2 + 1 \
                #         or pts_in_occlusion(occ_bbox, pts):
                flag = 0
                while pts_in_occlusion(occ_bbox, pts) and flag < 1000:
                    # occ_bbox = occlusion_single(true_bbox, occ_rate)
                    occ_bbox = occlusion_single(true_bbox, occ_rate)
                    flag += 1
                while pts_in_occlusion(occ_bbox, [point]):
                    # occ_bbox = occlusion_single(true_bbox, occ_rate)
                    occ_bbox = occlusion_single(true_bbox, occ_rate)

                ann['occ_bbox'] = occ_bbox
                n += 1
    # print(n)
    return coco
    # x1, y1, w, h = ann['true_bbox']
    #
    # x, y = ann['point']
    # rx, ry = (x - x1) / w, (y - y1) / h
    # X.append(rx)
    # Y.append(ry)
    # if (rx > 10) or ry > 10:
    #     print(rx, ry, x1, y1, w, h, x, y)
    # assert x1 - 1 < x < x1 + w + 1 and y1 - 1 < y < y1 + h + 1, f"{[x1, y1, x1 + w, y1 + h]} {x, y}"


def cal_union(occ_bbox, true_bbox):
    occ_bbox = x1y1wh_2_x1y1x2y2(occ_bbox)
    true_bbox = x1y1wh_2_x1y1x2y2(true_bbox)
    l, t = max(occ_bbox[0], true_bbox[0]), max(occ_bbox[1], true_bbox[1])
    r, b = min(occ_bbox[2], true_bbox[2]), min(occ_bbox[3], true_bbox[3])
    union = (r - l) * (b - t)
    return union


def check_occlusion(coco):
    n = 0
    b = 0
    final_occ_rate = 0
    real_occ_rate = 0
    imids = list(coco.imgs.keys())
    for id in imids:
        anns = coco.imgToAnns[id]
        num_gt = len(anns)
        pts = [ann['point'] for ann in anns]
        for ann in anns:
            true_bbox = ann['true_bbox']
            if true_bbox[2] > 1 and true_bbox[3] > 1:
                occ_bbox = ann['occ_bbox']
                point = ann['point']
                final_occ_rate += (occ_bbox[2] * occ_bbox[3]) / (true_bbox[2] * true_bbox[3])

                real_occ_rate += cal_union(occ_bbox, true_bbox) / (true_bbox[2] * true_bbox[3])
                print(final_occ_rate)
                print(occ_bbox)
                assert final_occ_rate < 1000000
                n += 1
                print(n)
                if pts_in_occlusion(x1y1wh_2_x1y1x2y2(occ_bbox), [point]):
                    print(x1y1wh_2_x1y1x2y2(occ_bbox))
                    print(pts)
                    b = b + 1
                    print(b)

    print('The final occ rate is:', final_occ_rate / n)
    print('The real occ rate is:', real_occ_rate / n)


ann_file = 'data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json'
output_fold = 'data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/occlusion/'
coco = GCOCO(ann_file)
occ_rate = 0.3
coco = occlusion(coco, occ_rate)
check_occlusion(coco)

json.dump(coco.dataset,
          open(output_fold + 'occlusion_{}_{}.json'.format(ann_file.split('/')[-1].split('.')[0], occ_rate), 'w'))
