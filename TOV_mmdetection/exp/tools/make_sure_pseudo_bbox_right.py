import json
from pycocotools.coco import COCO
import numpy as np


def valid(coarse_json, pseudo_json):
    coarse_jd = COCO(coarse_json)
    pseudo_jd = COCO(pseudo_json)
    for ann_id, coarse_ann in coarse_jd.anns.items():
        pseudo_ann = pseudo_jd.anns[ann_id]
        dis = (np.array(coarse_ann['point']) - np.array(pseudo_ann['point'])).sum()
        if dis > 1e-8:
            return False
    return True


for dist in ['noise_rg-0-0-0.25-0.25_1', 'noise_rg-0-0-0.167-0.167_1', 'noise_rg-0-0-0.125-0.125_1', 'noise_uniform_1']:
    print(dist, valid(
        coarse_json=f"data/coco/resize/coarse_annotations/{dist}/instances_train2017_100x167_coarse.json",
        pseudo_json=f"data/coco/resize/coarse_gen_annotations/{dist}/pseuw16h16/instances_train2017_100x167_coarse.json"
    ))
