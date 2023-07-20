import json
from copy import deepcopy
import numpy as np
import os
from huicv.corner_dataset.corner_utils import CocoAnnoUtil


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


class PointUtil(object):
    @staticmethod
    def crop(annos, corner, min_dis_to_edge=2):
        """
            corner: (l, t, r, b) which mean area of x in [l, r) and y in [t, b)
            1. find ann which point in corner image and keep it
            2. clip bbox and seg in keep anns
            3. translate ann to corner coordanate axies.
        """
        annos = deepcopy(annos)
        cx1, cy1, cx2, cy2 = corner
        # filter anno by area_keep_ratio
        points = np.array([anno['point'] for anno in annos])
        points -= np.array([cx1, cy1])
        keepx = (points[:, 0] >= min_dis_to_edge) & (points[:, 0] < cx2-cx1-min_dis_to_edge)
        keepy = (points[:, 1] >= min_dis_to_edge) & (points[:, 1] < cy2-cy1-min_dis_to_edge)
        keep = keepx & keepy
        annos = np.array(annos)[keep].tolist()

        if len(annos) > 0:
            for key in ['true_bbox', 'bbox']:
                if key in annos[0]:
                    CocoAnnoUtil.clip_bbox(annos, corner, key=key)
            if 'segmentation' in annos[0]:
                CocoAnnoUtil.clip_seg(annos, corner)

            keys = [key for key in ['point', 'true_bbox', 'bbox'] if key in annos[0]]
            if 'segmentation' in annos[0]:
                keys.append('seg')
            annos = CocoAnnoUtil.translate(annos, -cx1, -cy1, keys=keys, ignore_rle=True)

        return annos


def generate_corner_annos(image_info, annos, corner, corner_image_id, start_anno_id, ann_type='box', **clip_kwargs):
    """
    corner: [left, upper, right, lower], include left and upper, not include right and lower
    """
    corner_image_info = deepcopy(image_info)
    corner[2] = min([image_info['width'], corner[2]])
    corner[3] = min([image_info['height'], corner[3]])
    assert corner[0] < corner[2] and corner[1] < corner[3]
    corner_image_info['width'] = corner[2] - corner[0]
    corner_image_info['height'] = corner[3] - corner[1]
    corner_image_info['corner'] = corner
    corner_image_info['id'] = corner_image_id

    if len(annos) > 0:
        if ann_type == 'bbox':
            corner_annos = CocoAnnoUtil.crop(annos, corner, **clip_kwargs)  # will get deep copy one
        elif ann_type == 'point':
            corner_annos = PointUtil.crop(annos, corner, **clip_kwargs)
        else:
            raise ValueError(f'ann_type can only be "box" or "point", but got {ann_type}')
        for anno in corner_annos:
            anno['id'] = start_anno_id
            anno['image_id'] = corner_image_id
            start_anno_id += 1
    else:
        corner_annos = []
    return corner_image_info, corner_annos, corner


def generate_corner_annos_for_single_image(max_tile_size, tile_overlap, image_info, annos, corner_image_id,
                                           start_anno_id, log, ann_type, **clip_kwargs):
    w_over, h_over = tile_overlap
    w_tile, h_tile = max_tile_size

    annotations = []
    images_info = []
    H, W = image_info['height'], image_info['width']
    for cx1 in range(0, W, w_tile - w_over):
        for cy1 in range(0, H, h_tile - h_over):
            cx2 = cx1 + w_tile
            cy2 = cy1 + h_tile
            corner = [cx1, cy1, cx2, cy2]
            corner_image_info, corner_annos, corner = generate_corner_annos(
                image_info, annos, corner, corner_image_id, start_anno_id, ann_type, **clip_kwargs)
            if log:
                print("\tcorner image id: {}, corner: {}, len(annotations): {}".format(
                    corner_image_id, corner, len(corner_annos)))
            annotations.extend(corner_annos)
            images_info.append(corner_image_info)
            corner_image_id += 1
            start_anno_id += len(corner_annos)
    return {
        "images": images_info,
        "annotations": annotations,
    }


def generate_corner_dataset(ann_file, save_path=None, max_tile_size=(640, 640), tile_overlap=(100, 100), log=True,
                            ann_type='bbox', keep_origin_im_info=False, **clip_kwargs):
    """
        ann_file: str or dict, json annotation file or json annotation
        max_tile_size: (w, h)  sub image size
        tile_overlap: (w, h)
    """
    jd = json.load(open(ann_file)) if isinstance(ann_file, str) else ann_file

    corner_jd = {"images": [], "annotations": []}
    if keep_origin_im_info:
        corner_jd['old_images'] = deepcopy(jd['images'])

    if len(jd['annotations']) > 0:
        ann = jd['annotations'][0]
        assert ann_type in ann, f'corner ann_type specified as {ann_type}, but {ann_type} not in ann.'

    imid_to_annos, imid_to_images = group_anno_by_image_id(jd)
    for idx, imid in enumerate(imid_to_images):
        image_info = imid_to_images[imid]
        annos = imid_to_annos.get(imid, [])
        if log:
            print("[{}/{}]origin image_id: {}, (H, W): ({}, {}), len(annotations): {}".format(
                idx + 1, len(imid_to_images), imid, image_info['height'], image_info['width'], len(annos)))
        results = generate_corner_annos_for_single_image(
            max_tile_size, tile_overlap, image_info, annos, len(corner_jd['images']), len(corner_jd['annotations']),
            log, ann_type, **clip_kwargs)
        for key, value in results.items():
            corner_jd[key].extend(value)
        if log:
            print('\tall {} corner images, {} annotations in corner images'.format(
                len(results['images']), len(results['annotations'])))
    if log:
        print('total: {} corner images, {} annotations'.format(len(corner_jd['images']), len(corner_jd['annotations'])))

    for key in jd:
        if key not in corner_jd:
            corner_jd[key] = jd[key]

    if save_path is not None:
        save_dir, _ = os.path.split(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(corner_jd, open(save_path, 'w'), separators=(',', ':'))
        if log: print("generate corner dataset json file in {}".format(save_path))
    return corner_jd


# class RunOneTime(object):
#     """
#         run a code with multi-process or multi-thread, but part of code we may want only run for one time no
#         matter running in which process/thread
#     """
#     @staticmethod
#     def run(func, args=(), kwargs={}, lock_file='/tmp/_run_one_time.lock'):
#         import fcntl, os, threading
#         need_run = False
#         with open(lock_file, 'ra') as f:
#             fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # release after with block finished
#             lines = f.readline()
#             if len(lines) == 0:
#                 f.write("{}_{}".format(os.getpid(), threading.currentThread().getName()))
#                 need_run = True
#         if need_run:
#             func(*args, **kwargs)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('ann', help='ann_file')
    parser.add_argument('save_ann', help='save_ann', default='')
    parser.add_argument('--max_tile_w', help='', default=640,
                        type=int)
    parser.add_argument('--max_tile_h', help='', default=640,
                        type=int)
    parser.add_argument('--tile_overlap_w', help='', default=100,
                        type=int)
    parser.add_argument('--tile_overlap_h', help='', default=100,
                        type=int)
    parser.add_argument('--area_keep_ratio', help='threshold of left_area/origin_area of bbox to keep', default=0.3,
                        type=float)
    parser.add_argument('--size_th', help='min w/h bbox to keep', default=2, type=float)
    parser.add_argument('--area_th', help='min area bbox to keep', default=4, type=float)
    parser.add_argument('--log', help='', default=1, type=int)
    parser.add_argument('--ann_type', help='', default="bbox")
    parser.add_argument('--keep_origin_im_info', help='', default=0)
    args = parser.parse_args()

    ann_file = args.ann
    save_path = args.save_ann if len(args.save_ann) > 0 else None
    max_tile_size = (args.max_tile_w, args.max_tile_h)
    tile_overlap = (args.tile_overlap_w, args.tile_overlap_h)
    log = False if args.log == 0 else True
    ann_type = args.ann_type
    keep_origin_im_info = False if args.keep_origin_im_info == 0 else True

    clip_kwargs = dict(
        area_keep_ratio=args.area_keep_ratio,
        size_th=args.size_th,
        area_th=args.area_th
    )
    print(ann_file)
    # input()

    generate_corner_dataset(ann_file, save_path, max_tile_size, tile_overlap, log, ann_type, keep_origin_im_info,
                            **clip_kwargs)

    # clip_kwargs = dict(
    #     area_keep_ratio=0.3,  # cliped bbox's size and area > th will keep
    #     size_th=2,
    #     area_th=4)

    # ann_file = 'data/visDrone/coco_fmt_annotations/VisDrone2018-DET-train-person.json'
    # generate_corner_dataset(ann_file, save_path=None, max_tile_size=(640, 640), tile_overlap=(100, 100),
    #                         log=True, **clip_kwargs)
