from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

coco = COCO('../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/'
            'cascade_coarse_point_refine_r50_fpn_1x_coco400/incR2step3stage_c2c_loss0_r6_6_lr0.01_1x_8b8g/'
            'instances_train2017_refine2_2_r6_6.json')

# coco = COCO('data/coco/annotations/instances_train2017.json')
coco = COCO('data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json')


def add_segmentations(anns):
    for ann in anns:
        x1, y1, w, h = ann['bbox']
        x2, y2 = x1+w, y1+h
        ann['segmentation'] = [[x1, y1, x2, y1, x2, y2, x1, y2]]


def name2img(coco):
    return {imgs['file_name']: imgs for img_id, imgs in coco.imgs.items()}


name_to_img = name2img(coco)
img_info = name_to_img['000000415340.jpg']
img_id = img_info['id']
anns = coco.imgToAnns[img_id]
add_segmentations(anns)

print(img_info)
print(anns)
img_path = f"data/coco/images/{img_info['file_name']}"
img = np.array(Image.open(img_path))
plt.imshow(img)
coco.showAnns(anns, draw_bbox=True)
plt.show()
