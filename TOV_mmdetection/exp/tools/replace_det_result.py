import os
import json
from collections import defaultdict
import numpy as np
from huicv.coarse_utils.noise_data_utils import mean_ann_size_of_class


def keep_center_replace_wh(B, WH):
    B = np.array([B])
    WH = np.array([WH])
    W, H = WH[:, 0], WH[:, 1]
    xc, yc = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
    x1, x2 = xc - W / 2, xc + W / 2
    y1, y2 = yc - H / 2, yc + H / 2
    return np.stack([x1, y1, W, H], axis=1)[0].tolist()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("det")
    parser.add_argument("gt")
    args = parser.parse_args()

    gt_jd = json.load(open(args.gt))
    res = json.load(open(args.det))

    cls_mean_size = mean_ann_size_of_class(gt_jd, bbox_key='bbox')
    for det in res:
        det['bbox'] = keep_center_replace_wh(det['bbox'], cls_mean_size[det['category_id']])

    det_dir, det_name = os.path.split(args.det)
    json.dump(res, open(det_dir + '/' + 'cls_mean_size_' + det_name, 'w'))
