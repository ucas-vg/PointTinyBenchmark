from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from huicv.corner_dataset.corner_utils import CornerCOCO
import os.path as osp
import matplotlib.pylab as plt
from huicv.vis.visualize import draw_bbox, draw_center
from PIL import Image
from huicv.coarse_utils.point_utils.bbox_adjust import *
from huicv.json_dataset.coco_ann_utils import GCOCO
from pycocotools.coco import COCO
from mmdet.datasets.coco import CocoDataset
from huicv.coarse_utils.generate_new_bbox_json_file_corner import do_generate_class_wise


class Arg(object):
    ann = 'data/dota/DOTA-split/trainsplit/DOTA_train1024.json'
    img_root = 'data/dota/DOTA-split/trainsplit/images'
    config = 'configs2/DOTA/ECPL/ecpl_r50_fpn_1x_fl_sl1_DOTA_demo.py'
    checkpoint = 'work_dirs/ecpl_r50_fpn_1x_fl_sl1_DOTA/epoch_12.pth'
    device = 1
    score_th = 0.1


def main():
    # parser = ArgumentParser()
    # parser.add_argument('ann', help='ann_file')
    # parser.add_argument('img_root', help='img_root')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0., help='bbox score threshold')
    # args = parser.parse_args()

    args = Arg()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    dataset = GCOCO(args.ann)
    dataset.cat_ids = dataset.get_cat_ids(cat_names=CocoDataset.CLASSES)
    dataset.cat_id_to_cls_id = {cat_id: i for i, cat_id in enumerate(dataset.cat_ids)}

    img_ids = list(dataset.oriImgs.keys())
    iid = img_ids[5]
    anns = dataset.oriImgToAnns[iid]
    anns = [ann for ann in anns if not ann['ignore']]
    img_info = dataset.oriImgs[iid]
    img_path = osp.join(args.img_root, img_info['file_name'])

    result_filter = ResultFilter(num_per_gt=-1, score_th=args.score_th)
    cluster = MinDisCluster()
    # for visDronePerson 32
    # cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=2.0)]
    # for visDronePerson 16
    cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=1.25)]
    # cluster_filters.append(ContainPointFilter())
    cluster_filters = ClusterFilters(cluster_filters)
    box_gen = SimpleBoxGenerator('weight_mean', True)

    # test a single image
    results = inference_detector(model, img_path)

    print('ann cid:', set([ann['category_id'] for ann in anns]))
    results2 = []
    for cls_id, result in enumerate(results):
        cid = dataset.cat_ids[cls_id]
        if len(result) > 0:
            print('got cid', cid)
        for r in result.tolist():
            results2.append(r + [cid])
    results = results2
    print(len(results))
    bboxes, anns, old_anns = do_generate_class_wise(anns, results, result_filter, cluster,
                                                    cluster_filters, box_gen)

    # all_anns = anns
    # all_bboxes = []
    # for cat_id, anns in dataset.group_by_cat_id(all_anns).items():
    #     result = results[dataset.cat_id_to_cls_id[cat_id]]
    #     # show the results
    #     # show_result_pyplot(model, img_path, result, score_thr=args.score_thr)
    #     result = result_filter(anns, result)
    #     print(len(anns), len(result))
    #     clusters = cluster(anns, result)
    #     clusters = cluster_filters(anns, result, clusters)
    #     bboxes, idxes = box_gen(anns, result, clusters)
    #     all_bboxes.append(bboxes)
    # bboxes = np.concatenate(all_bboxes, axis=0)
    # anns = all_anns

    img = np.array(Image.open(img_path))
    plt.figure(figsize=(14, 8))
    plt.imshow(img)

    # show_cluster(anns, clusters, 3)
    # show_cluster(anns, clusters, 1)
    # show_cluster(anns, clusters, 5)
    # show_cluster(anns, clusters, 6)
    show_annos(anns, )
    # draw_bbox(plt.gca(), results, normalized_label=False)
    # draw_bbox(plt.gca(), bboxes, color=(1, 0, 0), normalized_label=False)
    # draw_bbox(plt.axes(), result, color=(1, 1, 0), normalized_label=False)

    draw_center(plt.gca(), result_filter(anns, np.array(results)), color='r')
    plt.show()


if __name__ == '__main__':
    """
    data/visDrone/coco_fmt_annotations/noise/corner/VisDrone2018-DET-train-person_noisept_corner_w640h640ow100oh100_pseuw32h32.json \
data/visDrone/VisDrone2018-DET-train/images/ \
configs/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE.py \
../mmdetection_cache/work_dir/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/pesu32_640_lr0.01_8e11e12e_2/epoch_12.pth \
    """
    main()
