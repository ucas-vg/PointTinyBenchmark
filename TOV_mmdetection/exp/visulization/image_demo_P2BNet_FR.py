import os
import json

from mmdet.apis import (inference_detector,
                        init_detector)


def get_img_list(ann_file, folder):
    imgs = []
    if ann_file:
        jd = json.load(open(ann_file))
        for i in jd['images']:
            imgs.append(i['file_name'])
    elif folder:
        imgs_raw = os.listdir(folder)
        for f in imgs_raw:
            if f.endswith(('jpg', 'png')):
                startIndex = f.find('_')
                endIndex = f.find('.')
                if startIndex > 0:
                    targetStr = f[startIndex:endIndex]
                    imgs.append(f.replace(targetStr, ''))
                else:
                    imgs.append(f)

    return imgs


# model
config_file = 'configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../TOV_mmdetection_cache/work_dir/coco/Faster/epoch_12.pth' # '../../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/detection/without_weight/epoch_12.pth'

# test a batch of images, show and save the results
ann_file = 'data/coco/annotations/instances_val2017.json'  # debug: only val set
in_folder = 'data/coco/images/'  # debug: only specific images
img_dir = 'data/coco/images/'
out_dir = '../TOV_mmdetection_cache/work_dir/coco/detection/visualization/'

imgs = get_img_list(None, in_folder)

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# i = 0
for f in imgs:
    # if i > 10000:
    #     break
    img = os.path.join(img_dir, f)
    result = inference_detector(model, img)
    # show_result_pyplot(model, img, result, score_thr=0.8)
    model.show_result(
        # api: mmdet.models.detectors.base.BaseDetector.show_result()
        img,
        result,
        score_thr=0.8,
        # show=True,
        show=False,
        wait_time=0,
        win_name='result',
        thickness=4,  # default 2
        font_size=24,
        bbox_color=(255, 190, 0),  # (72, 101, 241)
        text_color=(255, 190, 0),  # (72, 101, 241)
        out_file=os.path.join(out_dir, f))

    # i += 1
