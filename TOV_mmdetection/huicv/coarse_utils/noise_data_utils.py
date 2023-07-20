import numpy as np
import os.path as osp
import json
from copy import deepcopy
import os
from collections import defaultdict
from huicv.interactivate.path_utils import makedirs_if_not_exist
from huicv.interactivate.cmd import input_yes
from huicv.json_dataset.coco_ann_utils import dump_coco_annotation


def dump_json(jd, ann_file, success_info=None):
    if osp.exists(ann_file):
        if not input_yes(f"{ann_file} have exists, do you want to overwrite? (y/n)"):
            return False
    makedirs_if_not_exist(ann_file)
    dump_coco_annotation(jd, ann_file)
    if success_info is not None:
        print(success_info)
    return True


def group_anno_by_image_id(coco_train):
    imid_to_annos = {}
    for anno in coco_train['annotations']:
        if anno['image_id'] not in imid_to_annos:
            imid_to_annos[anno['image_id']] = [anno]
        else:
            imid_to_annos[anno['image_id']].append(anno)
    imid_to_images = {}
    for image_info in coco_train['images']:
        imid_to_images[image_info['id']] = image_info
    return imid_to_annos, imid_to_images


def clip_in_image(jd, keys=['bbox', 'seg']):
    """"""
    keys = deepcopy(keys)
    remove_seg = False
    if 'seg' in keys:
        remove_seg = True
        keys.remove('seg')

    from huicv.corner_dataset.corner_utils import CocoAnnoUtil
    imid_to_annos, imid_to_images = group_anno_by_image_id(jd)
    for iid in imid_to_annos:
        annos = imid_to_annos[iid]
        img_info = imid_to_images[iid]
        w, h = img_info['width'], img_info['height']
        ltrb = (0, 0, w, h)
        if remove_seg:
            CocoAnnoUtil.clip_seg(annos, ltrb)
        for key in keys:
            CocoAnnoUtil.clip_bbox(annos, ltrb, key)


def range_gaussian_sample(size, mu, sigma, a, b):
    assert isinstance(size, int)
    res = []
    c = 0
    while c < size:
        r = np.random.randn(2*(size-c)) * sigma + mu
        keep = (r >= a) & (r < b)
        res.append(r[keep])
        c += len(res[-1])
    return np.concatenate(res)[:size]


def add_rand_points(jd, size_range=1.0, rand_type='uniform', **rand_kwargs):
    # [-0.5, 0.5]
    if rand_type == 'uniform':
        rand_offset = np.random.rand(len(jd['annotations']), 2)
        rand_offset -= 0.5
    elif rand_type == 'range_gaussian':
        rand_offset = range_gaussian_sample(len(jd['annotations'])*2, a=-0.5, b=0.5, **rand_kwargs).reshape(-1, 2)
    else:
        raise ValueError("rand_type is not valid, only support ['uniform', 'range_gaussian']")
    print('mean, std', rand_offset.mean(axis=0), rand_offset.std(axis=0))
    bboxes = np.array([anno['bbox'] for anno in jd['annotations']])
    X1, Y1, W, H = bboxes.T
    Xc, Yc = X1 + W / 2, Y1 + H / 2
    x_off, y_off = rand_offset[:, 0] * size_range * W, rand_offset[:, 1] * size_range * H
    Xc += x_off
    Yc += y_off
    noise_point = np.stack((Xc, Yc), axis=-1).tolist()
    for i, anno in enumerate(jd['annotations']):
        anno['point'] = noise_point[i]


def replace_key_to_in_ann(jd, old_key, new_key):
    for ann in jd['annotations']:
        ann[new_key] = ann[old_key]
        del ann[old_key]


def xcycwh2x1y1x2y2(xc, yc, w, h):
    x1, x2 = xc - w / 2, xc + w / 2
    y1, y2 = yc - h / 2, yc + h / 2
    return np.stack((x1, y1, x2, y2), axis=-1)


def add_pseudo_bbox(jd, pseudo_wh=(32, 32), **clip_kwargs):
    """
        1. add pseudo bbox and delete segmentation
        2. clip box inside image
        3. filter small bbox
    """
    pseudo_wh = np.array(pseudo_wh)
    noise_points = np.array([anno['point'] for anno in jd['annotations']])
    if len(pseudo_wh.shape) == 1 and pseudo_wh.shape[0] == 2:
        WH = np.zeros(shape=(len(noise_points), 2))
        WH[:, :] = np.array(pseudo_wh)
    elif len(pseudo_wh.shape) == 2 and pseudo_wh.shape[0] == len(noise_points):
        WH = pseudo_wh
    else:
        raise ValueError("")

    X1Y1X2Y2 = xcycwh2x1y1x2y2(noise_points[:, 0], noise_points[:, 1], WH[:, 0], WH[:, 1])
    X1Y1WH = np.concatenate((X1Y1X2Y2[:, :2], WH), axis=-1).tolist()

    for i, anno in enumerate(jd['annotations']):
        anno['bbox'] = X1Y1WH[i]
        if 'segmentation' in anno:
            del anno['segmentation']

    clip_in_image(jd, keys=['bbox'])

    from huicv.corner_dataset.corner_utils import CocoAnnoUtil
    len_a = len(jd['annotations'])
    jd['annotations'] = CocoAnnoUtil.filter_small_bbox(jd['annotations'], **clip_kwargs)
    print(f"annotations count from {len_a} to {len(jd['annotations'])}", )


def get_new_json_file_path(ann_file, data_root, sub_dir_name, suffix):
    if data_root is not None:
        if not osp.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)
    ann_file = "{}_{}.json".format(ann_file[:-5], suffix)
    ann_dir, ann_file_name = osp.split(ann_file)
    if sub_dir_name is not None:
        sub_dir = osp.join(ann_dir, sub_dir_name)
    else:
        sub_dir = ann_dir
    ann_file = osp.join(sub_dir, ann_file_name)
    return ann_file, sub_dir


def generate_noisept_dataset(ann_file, save_ann=None, *args, **kwargs):
    import json
    jd = json.load(open(ann_file))
    add_rand_points(jd, *args, **kwargs)
    replace_key_to_in_ann(jd, 'bbox', 'true_bbox')

    if save_ann is None:
        if not osp.isabs(ann_file):
            ann_file = osp.abspath(ann_file)
        ann_file, ann_dir = get_new_json_file_path(ann_file, None, 'noise', 'noisept')
        if not osp.exists(ann_dir):
            import os
            os.makedirs(ann_dir)
    else:
        ann_file = save_ann
    dump_json(jd, ann_file, 'save noisept annotation file in {}'.format(ann_file))


def generate_fixedpt_dataset(ann_file, save_ann=None, *args, **kwargs):
    def add_fixed_points(jd, fixed_xy, **rand_kwargs):
        bboxes = np.array([anno['bbox'] for anno in jd['annotations']])
        X1, Y1, W, H = bboxes.T
        rx, ry = fixed_xy
        X, Y = X1 + rx * W, Y1 + ry * H
        points = np.stack([X, Y], axis=1).tolist()
        for i, anno in enumerate(jd['annotations']):
            anno['point'] = points[i]

    import json
    jd = json.load(open(ann_file))
    add_fixed_points(jd, *args, **kwargs)
    replace_key_to_in_ann(jd, 'bbox', 'true_bbox')
    dump_json(jd, save_ann, 'save noisept annotation file in {}'.format(save_ann))


def generate_pseudo_bbox_for_point(ori_ann_file, save_ann_path, **noise_kwargs):
    if 'clip_kwargs' not in noise_kwargs:
        noise_kwargs['clip_kwargs'] = {}
    jd = json.load(open(ori_ann_file))
    add_pseudo_bbox(jd, noise_kwargs['pseudo_wh'], **noise_kwargs['clip_kwargs'])
    dump_json(jd, save_ann_path, 'save pseudo bbox annotation file in {}'.format(save_ann_path))
    return jd


def generate_dataset_with_fixed_size_bbox(ann_file, save_file, fixed_wh=(32, 32), wh_ratio=(0.5, 0.5), **clip_kwargs):
    def fixed_bbox_wh(bbox, fixed_wh, ratio):
        x1, y1, w, h = bbox
        xc, yc = x1 + w * ratio[0], y1 + h * ratio[1]
        nw, nh = fixed_wh
        x1, y1 = xc - nw / 2, yc - nh / 2
        return [x1, y1, nw, nh]

    jd = json.load(open(ann_file))
    for ann in jd['annotations']:
        ann['true_bbox'] = ann['bbox']
        ann['bbox'] = fixed_bbox_wh(ann['bbox'], fixed_wh, wh_ratio)
        del ann['segmentation']

    clip_in_image(jd, keys=['bbox'])

    from huicv.corner_dataset.corner_utils import CocoAnnoUtil
    len_a = len(jd['annotations'])
    jd['annotations'] = CocoAnnoUtil.filter_small_bbox(jd['annotations'], **clip_kwargs)
    print(f"annotations count from {len_a} to {len(jd['annotations'])}", )
    dump_json(jd, save_file, 'save pseudo bbox annotation file in {}'.format(save_file))


def mean_ann_size_of_class(jd, bbox_key='true_bbox', use='size'):
    cls_sizes = defaultdict(list)

    for ann in jd['annotations']:
        x, y, w, h = ann[bbox_key]
        if w >= 2 and h >= 2:
            cls_sizes[ann['category_id']].append([w, h])

    cls_mean_size = {}
    for cls, sizes in cls_sizes.items():
        sizes = np.array(sizes)
        w, h = np.exp(np.mean(np.log(sizes), axis=0))
        if use == 'size':
            s = (w * h) ** 0.5
            cls_mean_size[cls] = [s, s]
        elif use == 'wh':
            cls_mean_size[cls] = [w, h]
        else:
            raise ValueError("")
        # print(cls, np.mean(sizes), np.exp(np.mean(np.log(sizes))))
    return cls_mean_size


def generate_mean_bbox_for_point(ori_ann_file, save_ann_path, **noise_kwargs):
    jd = json.load(open(ori_ann_file))
    cls_mean_size = mean_ann_size_of_class(jd)
    WH = np.array([cls_mean_size[ann['category_id']] for ann in jd['annotations']])
    add_pseudo_bbox(jd, WH, **noise_kwargs['clip_kwargs'])
    dump_json(jd, save_ann_path, 'save pseudo bbox annotation file in {}'.format(save_ann_path))
    return jd


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('task', help='task, can be "generate_dataset_with_fixed_size_bbox", "generate_noisept_dataset",'
                                     ' "generate_pseudo_bbox_for_point", "generate_mean_bbox_for_point",'
                                     ' "generate_fixedpt_dataset"')
    parser.add_argument('ann', help='ann_file')
    parser.add_argument('save_ann', help='save_ann', default='')
    parser.add_argument('--pseudo_w', help='pseudo_w, only for generate_dataset_with_fixed_size_bbox', default=32, type=int)
    parser.add_argument('--pseudo_h', help='pseudo_h, only for generate_dataset_with_fixed_size_bbox', default=32, type=int)
    parser.add_argument('--w_ratio', help='w_ratio, only for generate_dataset_with_fixed_size_bbox', default=0.5, type=float)
    parser.add_argument('--h_ratio', help='w_ratio, only for generate_dataset_with_fixed_size_bbox', default=0.5, type=float)
    parser.add_argument('--size_range', help='size_range, only for generate_noisept_dataset, 1.0 means in bounding box',
                        default=1.0, type=float)
    parser.add_argument('--rand_type', help='random method, only for generate_noisept_dataset', default='uniform')
    parser.add_argument('--range_gaussian_mu', help='range_gaussian_mu only used while rand_type is range_gaussian',
                        default=0, type=float)
    parser.add_argument('--range_gaussian_sigma', help='range_gaussian_sigma only used while rand_type is range_gaussian',
                        default=1, type=float)
    parser.add_argument('--fixed_x', help="only for generate_fixedpt_dataset, fixed point, range of [0, 1]", default=0.5, type=float)
    parser.add_argument('--fixed_y', help="only for generate_fixedpt_dataset, fixed point, range of [0, 1]", default=0.5, type=float)
    args = parser.parse_args()

    if args.task == "generate_dataset_with_fixed_size_bbox":
        generate_dataset_with_fixed_size_bbox(
            args.ann, args.save_ann,
            (args.pseudo_w, args.pseudo_h),
            wh_ratio=(args.w_ratio, args.h_ratio)
        )
    elif args.task == 'generate_noisept_dataset':
        # print(args)
        assert len(args.save_ann) > 0
        # python mmdet/datasets/noise_data_utils.py
        generate_noisept_dataset(
            args.ann,
            args.save_ann,
            size_range=args.size_range,
            rand_type=args.rand_type,
            mu=args.range_gaussian_mu,
            sigma=args.range_gaussian_sigma
        )
    elif args.task == "generate_pseudo_bbox_for_point":
        assert len(args.save_ann) > 0
        generate_pseudo_bbox_for_point(
            args.ann,
            args.save_ann,
            pseudo_wh=(args.pseudo_w, args.pseudo_h),
            clip_kwargs={'size_th': 2, 'area_th': 4}
        )
    elif args.task == "generate_mean_bbox_for_point":
        generate_mean_bbox_for_point(
            args.ann,
            args.save_ann,
            clip_kwargs={'size_th': 2, 'area_th': 4}
        )
    elif args.task == 'generate_fixedpt_dataset':
        generate_fixedpt_dataset(
            args.ann,
            args.save_ann,
            fixed_xy=(args.fixed_x, args.fixed_y)
        )
    else:
        raise TypeError('task must in ["generate_dataset_with_fixed_size_bbox", "generate_noisept_dataset",'
                        ' "generate_pseudo_bbox_for_point", "generate_fixedpt_dataset"]')

    # generate_noisept_dataset(
    #     'data/visDrone/coco_fmt_annotations/VisDrone2018-DET-train-person.json',
    #     size_range=args.size_range
    # )

    # generate_noisept_dataset(
    #     'data/tiny_set/mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
    #     size_range=1.0
    # )

    # generate_noisept_dataset(
    #     'data/tiny_set/mini_annotations/tiny_set_train_all_erase.json',
    #     size_range=1.0
    # )

    # generate_dataset_with_fixed_size_bbox(
    #     'data/visDrone/coco_fmt_annotations/VisDrone2018-DET-train-person.json',
    #     'data/visDrone/coco_fmt_annotations/fixed_point/VisDrone2018-DET-train-person_pseuwh32_wh1d6ratio.json',
    #     (32, 32),
    #     wh_ratio=(1.0 / 6, 1.0 / 6)
    # )
    #
    # generate_dataset_with_fixed_size_bbox(
    #     'data/visDrone/coco_fmt_annotations/VisDrone2018-DET-train-person.json',
    #     'data/visDrone/coco_fmt_annotations/fixed_point/VisDrone2018-DET-train-person_pseuwh32_w1d2h5d6ratio.json',
    #     (32, 32),
    #     wh_ratio=(1.0 / 2, 5.0 / 6)
    # )

    # generate_dataset_with_fixed_size_bbox(
    #     'data/visDrone/coco_fmt_annotations/VisDrone2018-DET-train-person.json',
    #     'data/visDrone/coco_fmt_annotations/fixed_point/VisDrone2018-DET-train-person_pseuwh32_w1d2h1d6ratio.json',
    #     (32, 32),
    #     wh_ratio=(1.0 / 2, 1.0 / 6)
    # )
