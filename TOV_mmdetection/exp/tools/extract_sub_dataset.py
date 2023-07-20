from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy


def x1y1wh2xcycwh(x1y1wh):
    x1, y1, w, h = x1y1wh
    xc, yc = x1 + w/2, y1 + h/2
    return xc, yc, w, h


def im_name2im(coco):
    jd = coco.dataset
    name2im = {}
    for im_info in jd['images']:
        name2im[im_info['file_name']] = im_info
    return name2im


def find_imgs_id(names, name2im):
    imgs_id = []
    for name in names:
        if name not in name2im:
            print(name, 'is not in dataset, take like', list(name2im.keys())[0])
        imgs_id.append(name2im[name]['id'])
    return imgs_id


def get_sub_dataset(coco, imgs_id, pts_dict):
    anns = []
    for im_id in imgs_id:
        filename = coco.imgs[im_id]['file_name']
        anns.extend(modify_points(coco.imgToAnns[im_id], pts_dict[filename]))
    imgs = []
    for im_id in imgs_id:
        imgs.append(coco.imgs[im_id])

    data = {
        'images': imgs,
        'annotations': anns,
    }
    for key in coco.dataset:
        if key not in ['images', 'annotations']:
            data[key] = coco.dataset[key]

    return data


def show_img_by_id(img_id, sub_jd):
    for im_info in sub_jd['images']:
        if im_info['id'] == img_id:
            break

    anns = []
    for ann in sub_jd['annotations']:
        if ann['image_id'] == img_id:
            anns.append(ann)

    img = Image.open('data/coco/images/' + im_info['file_name'])
    plt.imshow(img)

    # anns = coco.imgToAnns[img_id]
    for ann in anns:
        px, py, _, _ = x1y1wh2xcycwh(ann['bbox'])
        plt.scatter(px, py)
    plt.show()


def modify_points(anns, all_pts):
    all_anns = []
    for i, old_ann in enumerate(anns):
        x1, y1, w, h = old_ann['bbox']
        if all_pts is None:
            all_anns.append(old_ann)
            continue
        for pts in all_pts:
            x, y = pts[i]
            ann = deepcopy(old_ann)
            ann['bbox'][0], ann['bbox'][1] = x - w / 2, y - h / 2
            all_anns.append(ann)
    return all_anns


coco = COCO('data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json')
name2im = im_name2im(coco)
imgs_id = find_imgs_id(['000000074711.jpg'], name2im)

sub_jd = get_sub_dataset(coco, imgs_id, {
    '000000074711.jpg':  # None
        [
            # [(229, 168), (423, 275)],
            # [(342, 213), (123, 192)],
            [(234, 132), (423, 275)],
         ]
})
json.dump(sub_jd, open(
    'data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/sub_4_instances_train2017_coarse.json', 'w'))

show_img_by_id(imgs_id[0], sub_jd)
