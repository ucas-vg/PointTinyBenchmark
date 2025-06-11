from huicv.corner_dataset.corner_utils import CornerCOCO
from huicv.vis.visualize import draw_bbox, draw_center, get_hsv_colors
from PIL import Image
from huicv.coarse_utils.point_utils.bbox_adjust import *
import os.path as osp


class Args:
    ann = "data/coco/resize/annotations/instances_val2017_100x167.json"
    img_root = "data/coco/resize/images_100x167_q100"
    score_th = 0.2


args = Args()
dataset = CornerCOCO(args.ann)

res = {}
for key in [
    ('uniform', 'latest_result_tinycoco_coarse3022.json'),
    ('center', 'latest_result_tinycoco_coarse3901.json'),
    ('rg0.0-0.25', 'latest_result_tinycoco_corase_3024.json')
]:
    res[key[0]] = dataset.parse_results(f"exp/{key[1]}")

# img_ids = list(dataset.oriImgs.keys())
cate_name2id = {cate['name']: cate['id'] for cate in dataset.coco.dataset['categories']}
img_ids = dataset.coco.getImgIds(catIds=cate_name2id['bus'])

iid = img_ids[1]
anns = dataset.cornerImgToCornerAnns[iid]
anns = [ann for ann in anns if (not ann['ignore']) and (not ann['iscrowd'])]
img_info = dataset.cornerImgs[iid]
img_path = osp.join(args.img_root, img_info['file_name'])

img = np.array(Image.open(img_path))
bboxes = np.concatenate([res[key][iid] for key in ['uniform', 'center', 'rg0.0-0.25']], axis=0)

cate_id = {ann['category_id'] for ann in anns} | {b[-1] for b in bboxes}

colors = {cid: c for c, cid in zip(get_hsv_colors(len(cate_id)), cate_id)}

for key in ['center', 'uniform']:
    plt.figure(figsize=(14, 8))
    plt.imshow(img)

    for ann in anns:
        show_annos([ann], color=colors[ann['category_id']])

    bboxes = res[key][iid]
    bboxes = bboxes[bboxes[:, 4] > args.score_th]
    color = [colors[int(c)] for c in bboxes[:, -1]]
    # draw_bbox(plt.gca(), bboxes, color=(1, 0, 0), normalized_label=False)
    draw_center(plt.gca(), bboxes, color=color, s=(bboxes[:, 4]**2)*200)
    # draw_bbox(plt.gca(), result, color=(1, 1, 0), normalized_label=False)
    plt.title(key)
    plt.show()
