import os


def get_dataset_name_path(dconfig, data_dict=None):
    if data_dict is None:
        data_dict ={}
    for S in dconfig:
        C = S["corner"]
        if S["round"] is None:
            S["round"] = [None]
        for r in S["round"]:
            dname = "VisDrone2018-DET-train-person{}{}{}{}".format(
                "" if S["noise"] is None else ("_noisept" if S["noise"] == "uniform" else f"_{S['noise']}_noisept"),
                f"_corner_w{C[0]}h{C[1]}ow{C[2]}oh{C[3]}" if C is not None else "",
                "" if S['method'] is None else f"_{S['method']}",
                "" if r is None else f"_round{r}"
            )
            image_root = "visDrone/VisDrone2018-DET-train/images"
            json_file = "visDrone/coco_fmt_annotations{}{}{}/{}.json".format(
                        "" if S["noise"] is None else "/noise",
                        "" if S["corner"] is None else "/corner",
                        "" if r is None else "/round",
                        dname
                    )
#             assert os.path.exists("datasets/" + image_root), "datasets/{} not exist".format(image_root)
#             assert os.path.exists("datasets/" + json_file), "datasets/{} not exist".format(json_file)
            data_dict[dname] = (image_root, json_file)
    return data_dict


def get_json_file(S, ann_root='coco_fmt_annotations/'):
    C = S["corner"]
    r = S['round']
    dname = "VisDrone2018-DET-train-person{}{}{}{}".format(
        "" if S["noise"] is None else ("_noisept" if S["noise"] == "uniform" else f"_{S['noise']}_noisept"),
        f"_corner_w{C[0]}h{C[1]}ow{C[2]}oh{C[3]}" if C is not None else "",
        "" if S['method'] is None else f"_{S['method']}",
        "" if r is None else f"_round{r}"
    )

    json_file = "{}{}{}{}/{}.json".format(
        ann_root,
        "" if S["noise"] is None else "/noise",
        "" if S["corner"] is None else "/corner",
        "" if r is None else "/round",
        dname
    )
    return json_file


import numpy as np


def get_centerwh(x1y1wh):
    x1, y1, w, h = x1y1wh
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]


def get_point_positions(anno_data):
    positions = []
    WH = []
    for ann_id, ann in anno_data.anns.items():
        true_box = get_centerwh(np.array(ann['true_bbox']))
        if true_box[-1] > 0 and true_box[-2] > 0:
            point = np.array(get_centerwh(ann['bbox'])[:2] if 'bbox' in ann else ann['point'])
            pos = (point - true_box[:2]) / true_box[2:]
            positions.append(pos)
            WH.append(true_box[2:])
        else:
            print("[ignore]:", ann)
    positions = np.array(positions)
    WH = np.array(WH)
    return np.concatenate([positions, WH], axis=-1)


def deal_invalid_position(positions, method='filter', bound=(-0.5, 0.5)):
    ps = positions[:, :2]
    assert bound[0] < bound[1]
    keep = (((ps > bound[1]) | (ps < bound[0])).sum(axis=-1) == 0)
    print("invalid/all={}/{}".format(len(positions)-keep.sum(), len(positions)))
    if method == 'filter':
        return positions[keep]
    elif method == 'clip':
        positions[:, :2] = positions[:, :2].clip(-0.5, 0.5)
        return positions
    else:
        raise ValueError()


def size_filter(positions, size_th):
    keep = (positions[:, 2] * positions[:, 3] >= size_th[0] * size_th[0]) & (positions[:, 2] * positions[:, 3] < size_th[1] * size_th[1])
    return positions[keep]


#


class Obj(object):
    def __init__(self, anns):
        self.anns = anns


def get_moved_point(anno_data1, anno_data2):
    new_anno_data = {}
    for ann_id in anno_data1.anns:
        ann1 = anno_data1.anns[ann_id]
        ann2 = anno_data2.anns[ann_id]
        if ann1['bbox'] != ann2['bbox']:
            new_anno_data[ann_id] = ann2
    return Obj(new_anno_data)


def get_keep_point(anno_data1, anno_data2):
    new_anno_data = {}
    for ann_id in anno_data1.anns:
        ann1 = anno_data1.anns[ann_id]
        ann2 = anno_data2.anns[ann_id]
        if ann1['bbox'] == ann2['bbox']:
            new_anno_data[ann_id] = ann2
    return Obj(new_anno_data)


#


def group_by_origin_image_name(jd):
    origin_name_to_images = {}
    for image_info in jd['images']:
        file_name = image_info['file_name']
        if file_name not in origin_name_to_images:
            origin_name_to_images[file_name] = [image_info]
        else:
            origin_name_to_images[file_name].append(image_info)
    return origin_name_to_images
