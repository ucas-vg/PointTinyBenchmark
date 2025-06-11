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
    config = 'configs2/DOTA/ECPL/ecpl_r50_fpn_1x_fl_sl1_DOTA_demo.py'
    checkpoint = 'work_dirs/ecpl_r50_fpn_1x_fl_sl1_DOTA/epoch_12.pth'
    img_path = 'data/dota/DOTA-split/trainsplit/images'
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

    # test a single image
    results = inference_detector(model, args.img_path)

    img = np.array(Image.open(args.img_path))
    plt.figure(figsize=(14, 8))
    plt.imshow(img)

    draw_bbox(plt.gca(), results, normalized_label=False)
    # draw_bbox(plt.gca(), bboxes, color=(1, 0, 0), normalized_label=False)
    # draw_bbox(plt.axes(), result, color=(1, 1, 0), normalized_label=False)
    plt.show()


if __name__ == '__main__':
    """
    data/visDrone/coco_fmt_annotations/noise/corner/VisDrone2018-DET-train-person_noisept_corner_w640h640ow100oh100_pseuw32h32.json \
data/visDrone/VisDrone2018-DET-train/images/ \
configs/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE.py \
../mmdetection_cache/work_dir/locnet/noisept1/reppoints_moment_r50_fpn_gn-neck+head_1x_visDroneNoisept640_bboxE/pesu32_640_lr0.01_8e11e12e_2/epoch_12.pth \
    """
    main()
