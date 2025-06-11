from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from huicv.corner_dataset.corner_utils import CornerCOCO
import os.path as osp
import matplotlib.pylab as plt
from huicv.vis.visualize import draw_bbox
from PIL import Image
from huicv.coarse_utils.point_utils.bbox_adjust import *


def main():
    parser = ArgumentParser()
    parser.add_argument('ann', help='ann_file')
    parser.add_argument('img_root', help='img_root')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0., help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    dataset = CornerCOCO(args.ann)
    img_ids = list(dataset.oriImgs.keys())
    iid = img_ids[0]
    anns = dataset.oriImgToAnns[iid]
    anns = [ann for ann in anns if not ann['ignore']]
    img_info = dataset.oriImgs[iid]
    img_path = osp.join(args.img_root, img_info['file_name'])

    result_filter = ResultFilter(num_per_gt=-1, score_th=0.0)
    cluster = MinDisCluster()
    # for visDronePerson 32
    # cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=2.0)]
    # for visDronePerson 16
    cluster_filters = [TopkFilter(16), GrowRateFilter(max_grow_rate=1.5)]
    # cluster_filters.append(ContainPointFilter())
    cluster_filters = ClusterFilters(cluster_filters)
    box_gen = SimpleBoxGenerator('weight_mean', True)

    # test a single image
    result = inference_detector(model, img_path)
    result = result[0]

    # show the results
    # show_result_pyplot(model, img_path, result, score_thr=args.score_thr)
    result = result_filter(anns, result)
    print(len(anns), len(result))
    clusters = cluster(anns, result)
    clusters = cluster_filters(anns, result, clusters)
    bboxes, idxes = box_gen(anns, result, clusters)
    img = np.array(Image.open(img_path))
    plt.figure(figsize=(14, 8))
    plt.imshow(img)

    # show_cluster(anns, clusters, 3)
    # show_cluster(anns, clusters, 1)
    # show_cluster(anns, clusters, 5)
    # show_cluster(anns, clusters, 6)
    show_annos(anns, )

    draw_bbox(plt.gca(), bboxes, color=(1, 0, 0), normalized_label=False)
    # draw_bbox(plt.gca(), result, color=(1, 1, 0), normalized_label=False)

    plt.show()


if __name__ == '__main__':
    """
    data/visDrone/coco_fmt_annotations/noise/corner/VisDrone2018-DET-train-person_noisept_corner_w640h640ow100oh100_pseuw32h32.json \
data/visDrone/VisDrone2018-DET-train/images/ \
configs/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE.py \
../mmdetection_cache/work_dir/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/pesu32_640_lr0.01_8e11e12e_2/epoch_12.pth \
    """
    main()
