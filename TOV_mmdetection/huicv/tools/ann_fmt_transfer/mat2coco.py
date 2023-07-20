import scipy.io as scio
import numpy as np
import os
from huicv.tools.ann_fmt_transfer.utils import *
from PIL import Image


def _parse_mat(fpath, img_id, anno_id):
    loader = scio.loadmat(fpath)
    #     print(loader['image_info'].shape, loader['image_info'][0, 0].shape,  loader['image_info'][0, 0][0, 0].shape)
    #     print(loader['image_info'][0, 0][0, 0][0].shape)
    #     loader['image_info'][0, 0][0, 0]
    pts = loader['image_info'][0, 0][0, 0][0]
    bboxs = _to_pseudo_box(pts)
    annos, anno_id = _to_annos(anno_id, img_id, bboxs)
    return annos, anno_id


def _to_pseudo_box(pts, wh=(15, 15)):
    #     wh = np.array(wh)
    #     x1y1 = pts - wh / 2
    #     x2y2 = pts + wh / 2
    #     WH = np.zeros(pts.shape) + wh
    #     return np.concatenate((x1y1, WH), axis=1)
    wh = np.array(wh)
    x1y1 = pts - wh / 2
    x2y2 = pts + wh / 2
    return np.concatenate((x1y1, x2y2), axis=1)


def _to_annos(anno_id, img_id, bboxs):
    annos = []
    for box in bboxs:
        ann = fmt_coco_annotation(box, 1, anno_id, img_id)  # category_id = 1
        annos.append(ann)
        anno_id += 1
    return annos, anno_id


def _fmt_image(file_name, height, width, img_id):  # f"{mode}/{file_name}",
    return {'file_name': file_name,
            'height': height,
            'width': width,
            'id': img_id}


def generate_coco_fmt(onefolder, categories, img_onefolder):
    img_id, anno_id = 0, 0
    all_annos, all_images = [], []

    for f in os.listdir(onefolder):
        if f.endswith('.mat'):
            annos, anno_id = _parse_mat(os.path.join(onefolder, f), img_id, anno_id)
            all_annos.extend(annos)

            name = f[f.find('_') + 1: f.find('.')]
            name = f"{name}.jpg"
            img = Image.open(os.path.join(img_onefolder, name))
            all_images.append(_fmt_image(name, img.height, img.width, img_id))
            img_id += 1

    return {
        "images": all_images,
        "annotations": all_annos,
        "categories": categories,
        "type": "instance"
    }


if __name__ == '__main__':
    categories = [{'id': 1, 'name': 'person', 'supercategory': 'person'}]
    mat_dir = ''
    img_dir = ''
    generate_coco_fmt(mat_dir, categories, img_dir)
