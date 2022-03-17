import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

# ###########################didi##################################################################################
import mmcv
import numpy as np
import matplotlib.pyplot as plt


def show_result_p2p(model, img, result, score_thr=0.3):
    """
    Args:
        model:
        img: 'data/coco/images/000000062060.jpg'
        result:
        score_thr:
    Returns:
    """

    def imshow_det_pts(img,
                       bboxes,  # (36,5)
                       labels,  # (36,)
                       score_thr=0,
                       ):
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        EPS = 1e-2

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)
        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        pts = pseudo_bbox_to_center(bboxes[:, :4])
        pts = pts.astype(np.int32)
        # for i, (pt, label) in enumerate(zip(pts, labels)):
        #     pt_int = pt.astype(np.int32)

        np.random.seed(0)
        color = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(80 + 1)
        ]
        colors = [color[labels[i]] for i in range(labels.shape[0])]
        plt.scatter(pts[:, 0], pts[:, 1], c=np.squeeze(colors)/256)
        # color = []
        plt.imshow(img)
        # plt.show()
        plt.clf()
        plt.savefig("")

    def pseudo_bbox_to_center(gt_bboxes):
        return (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

    img = mmcv.imread(img).astype(np.uint8)

    bboxes = np.vstack(result)  # {array: (n,5)}
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    # if out_file specified, do not show image in window
    # draw bounding boxes
    imshow_det_pts(
        img,
        bboxes,
        labels,
        score_thr
    )
# ###########################didi##################################################################################

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--save-dir', default='exp/debug/P2P/coarse/', help='')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    show_result_p2p(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
