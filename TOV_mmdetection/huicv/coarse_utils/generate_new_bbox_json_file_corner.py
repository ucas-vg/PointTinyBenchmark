from argparse import ArgumentParser

from huicv.corner_dataset.corner_utils import CornerCOCO, CocoAnnoUtil
import os.path as osp
import matplotlib.pylab as plt
from huicv.vis.visualize import draw_bbox
from PIL import Image
from huicv.coarse_utils.point_utils.bbox_adjust import *
import json
from copy import deepcopy
from huicv.json_dataset.coco_ann_utils import dump_coco_annotation


def dump_new_json_file(dataset, save_path):
    annotations = []
    for corner_img in dataset.cornerImgs.values():
        annos = dataset.cornerImgToCornerAnns[corner_img['id']]
        if len(annos) == 0: continue
        corner = (0, 0, corner_img['width'], corner_img['height'])
        CocoAnnoUtil.clip_bbox(annos, corner, key='bbox')
        annos = CocoAnnoUtil.filter_small_bbox(annos)
        annotations.extend(annos)

    print(f"annotation from {len(dataset.coco.dataset['annotations'])} to {len(annotations)}")
    jd = dict(annotations=annotations)
    for key in dataset.coco.dataset:
        if key not in jd:
            jd[key] = dataset.coco.dataset[key]

    for ann in jd['annotations']:
        if 'is_fully' in ann and ann['is_fully']:
            ann['bbox'] = ann['true_bbox']
    dump_coco_annotation(jd, save_path)


def do_generate_class_wise(all_anns, all_result,
                           result_filter, cluster, cluster_filters, box_gen):
    all_bboxes, all_old_anns, all_new_anns, all_no_change_anns = [], [], [], []

    cid2anns = CornerCOCO.group_anns_by_cat_id(all_anns)
    cid2results = CornerCOCO.group_results_by_cat_id(all_result)
    for cid, anns in cid2anns.items():
        if cid not in cid2results:
            all_no_change_anns.extend(anns)
            continue
        result = cid2results[cid]
        if len(result) == 0:
            pass

        result = result_filter(anns, result)
        clusters = cluster(anns, result)
        clusters = cluster_filters(anns, result, clusters)
        bboxes, idxes = box_gen(anns, result, clusters)

        ann_idx2_bbox_idx = {ann_idx: i for i, ann_idx in enumerate(idxes)}
        for ann_idx in range(len(anns)):
            if ann_idx not in ann_idx2_bbox_idx:
                all_no_change_anns.append(anns[ann_idx])
                continue
            bbox_idx = ann_idx2_bbox_idx[ann_idx]

            bbox, corner_ann, old_ann = bboxes[bbox_idx], anns[ann_idx], deepcopy(anns[ann_idx])
            x1, y1, x2, y2 = bbox.tolist()
            corner_ann['bbox'] = [x1, y1, x2 - x1, y2-y1]

            all_bboxes.append(bbox)
            all_old_anns.append(old_ann)
            all_new_anns.append(corner_ann)

    all_bboxes = np.array(all_bboxes)
    all_old_anns.extend(all_no_change_anns)
    all_new_anns.extend(all_no_change_anns)
    return all_bboxes, all_new_anns, all_old_anns


def main():
    parser = ArgumentParser()
    parser.add_argument('ann_file', help='ann_file')
    parser.add_argument('result_file', help='result_file')
    parser.add_argument('img_root', help='img_root')
    parser.add_argument('save_path', help='save_path')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--max-grow-rate', type=float, default=1.5, help='bbox score threshold')
    parser.add_argument('--show', dest='show', help='cluster_topk', default=1, type=int)
    args = parser.parse_args()
    show = True if args.show != 0 else False
    dataset = CornerCOCO(args.ann_file)
    cornerimgid2results = CornerCOCO.parse_results(args.result_file)
    # imgname2results = CornerCOCO.get_imgname2results(args.result_file, args.result_gt_file)
    print('load image over.')

    # result_filter = ResultFilter(num_per_gt=-1, score_th=0.2)
    result_filter = ResultFilter(num_per_gt=-1, score_th=0.)
    cluster = MinDisCluster()
    # cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=2)]
    cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=args.max_grow_rate)]
    # cluster_filters.append(ContainPointFilter())
    cluster_filters = ClusterFilters(cluster_filters)
    box_gen = SimpleBoxGenerator('weight_mean', True)

    # # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)

    img_ids = list(dataset.cornerImgs.keys())
    for it, iid in enumerate(img_ids):
        anns = dataset.cornerImgToCornerAnns[iid]
        if len(anns) == 0:
            continue
        img_info = dataset.cornerImgs[iid]
        imgname = img_info['file_name']
        if iid not in cornerimgid2results: continue
        result = cornerimgid2results[iid]

        bboxes, anns, old_anns = do_generate_class_wise(anns, result, result_filter, cluster,
                                                        cluster_filters, box_gen)

        if show:
            img_path = osp.join(args.img_root, imgname)
            img = np.array(Image.open(img_path).crop(img_info['corner']))

            plt.imshow(img)

            # print(idxes)
            a = np.array([ann['bbox'] for ann in old_anns])
            a[:, [0, 1]] += a[:, [2, 3]] / 2
            b = np.array(bboxes)
            b[:, [0, 1]] = (b[:, [0, 1]] + b[:, [2, 3]]) / 2
            # print(a[:, [0, 1]] - b[:, [0, 1]])

            show_annos(old_anns)
            # draw_bbox(plt.axes(), bboxes[:, :4], color=(1, 0, 0), normalized_label=False)
            # draw_bbox(plt.axes(), result[:, :4], color=(1, 0, 0), normalized_label=False)
            plt.show()
            x = input()
            if x == '0':
                break
            elif x == '1':
                show = False
    if not show:
        dump_new_json_file(dataset, args.save_path)
        # dump_new_json_file(dataset,
        #                    'data/visDrone/coco_fmt_annotations/noise/corner/round/'
        #                    'VisDrone2018-DET-train-person_noisept_corner_w640h640ow100oh100_2_round2.json')
    # plt.show()


if __name__ == '__main__':
    """
    data/visDrone/coco_fmt_annotations/noise/corner/VisDrone2018-DET-train-person_noisept_corner_w640h640ow100oh100_pseuw32h32.json
    exp/latest_result.json
    data/visDrone/coco_fmt_annotations/noise/VisDrone2018-DET-train-person_noisept.json
    data/visDrone/VisDrone2018-DET-train/images
    """
    main()
