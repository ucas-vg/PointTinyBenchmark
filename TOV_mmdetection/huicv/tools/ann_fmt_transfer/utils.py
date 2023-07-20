def fmt_coco_annotation(bbox, cid, anno_id, img_id, iscrowd=0, ignore=None):
    x1, y1, x2, y2 = bbox
    ann = {'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'category_id': cid,
            'area': (y2 - y1) * (x2 - x1),
            'iscrowd': iscrowd,
            'image_id': img_id,
            'id': anno_id}

    if ignore is not None:
        ann['ignore'] = ignore
    return ann



