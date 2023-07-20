import json
import numpy as np


def filter_ignore_out(coco_fmt):
    coco_fmt['annotations'] = [anno for anno in coco_fmt['annotations'] if 'ignore' not in anno or not anno['ignore']]


def filter_small_bbox(coco_fmt):
    coco_fmt['annotations'] = [anno for anno in coco_fmt['annotations'] if anno['bbox'][2] > 1 and anno['bbox'][3] > 1]


# absolute-size
def get_size(annos, *args, **kwargs):
    rgb_bboxes = [anno['bbox'] for anno in annos['annotations']]
    rgb_bboxes = np.array(rgb_bboxes)
    sizes = np.sqrt(rgb_bboxes[:, 2] * rgb_bboxes[:, 3])
    return sizes


# relative-sizes
def get_rsize(annos, iid_to_img, *args, **kwargs):
    rgb_bboxes = [anno['bbox'] for anno in annos['annotations']]
    rgb_bboxes = np.array(rgb_bboxes)
    rgb_imgs = [iid_to_img[anno['image_id']] for anno in annos['annotations']]
    imgs_sizes = [img['height'] * img['width'] for img in rgb_imgs]
    imgs_sizes = np.array(imgs_sizes)
    sizes = np.sqrt((rgb_bboxes[:, 2] * rgb_bboxes[:, 3]) / imgs_sizes[:])
    return sizes


def get_aspect(annos, *args, **kwargs):
    rgb_bboxes = [anno['bbox'] for anno in annos['annotations']]
    rgb_bboxes = np.array(rgb_bboxes)
    return rgb_bboxes[:, 2] / rgb_bboxes[:, 3]


def print_statistic(rgb_jd, rtrain_jd, rvalid_jd, rtest_jd):
    if rgb_jd is None:
        rgb_jd = {'images': [], 'annotations': []}
        for jd in [rtrain_jd, rvalid_jd, rtest_jd]:
            rgb_jd['images'].extend(jd['images'])
            rgb_jd['annotations'].extend(jd['annotations'])

    attr = 'images'
    print(len(rtrain_jd[attr]), len(rvalid_jd[attr]), len(rtest_jd[attr]), len(rgb_jd[attr]))
    attr = 'annotations'
    print(len(rtrain_jd[attr]), len(rvalid_jd[attr]), len(rtest_jd[attr]), len(rgb_jd[attr]))

    # 建立映射
    rgb_iid_to_img, xray_iid_to_img = {}, {}
    for img in rgb_jd['images']:
        rgb_iid_to_img[img['id']] = img

    filter_ignore_out(rgb_jd)  # 过滤掉['ignore']＝＝True的annos
    filter_small_bbox(rgb_jd)
    rgb_bboxes = get_size(rgb_jd)
    # print("[%.1f, %.1f]"%(np.min(rgb_bboxes), np.max(rgb_bboxes)))
    print("[%d, %d]" % (np.min(rgb_bboxes), np.max(rgb_bboxes)))  # 截断取整！！
    print("absolute size: %.3f±%.3f" % (np.mean(rgb_bboxes), np.std(rgb_bboxes)))
    rgb_rbboxes = get_rsize(rgb_jd, rgb_iid_to_img)
    print("relative size: %.3f±%.3f" % (np.mean(rgb_rbboxes), np.std(rgb_rbboxes)))
    rgb_aspects = get_aspect(rgb_jd)
    print("aspect ratio: %.3f±%.3f" % (np.mean(rgb_aspects), np.std(rgb_aspects)))
    print()


root_path = 'data/tiny_set_v2/'
# rgb_jd = json.load(open(root_path+'anns/origin/rgb_all.json'))
rtrain_jd = json.load(open(root_path+'anns/origin/rgb_train.json'))
rvalid_jd = json.load(open(root_path+'anns/origin/rgb_valid.json'))
rtest_jd = json.load(open(root_path+'anns/origin/rgb_test.json'))

# print_statistic(rgb_jd, rtrain_jd, rvalid_jd, rtest_jd)
print_statistic(None, rtrain_jd, rvalid_jd, rtest_jd)
"""
5711 568 5753 12032
262063 42399 315165 619627
[2, 309]
absolute size: 22.619±10.849
relative size: 0.016±0.007
aspect ratio: 0.723±0.424
"""


root_path = 'data/tiny_set/'
# rgb_jd = json.load(open(root_path+'anns/origin/rgb_all.json'))
rtrain_jd = json.load(open(root_path+'annotations/tiny_set_train_with_dense.json'))
rvalid_jd = json.load(open(root_path+'annotations/tiny_set_test_with_dense.json'))
# rtest_jd = json.load(open(root_path+'annotations/origin/rgb_test.json'))

# print_statistic(rgb_jd, rtrain_jd, rvalid_jd, rtest_jd)
print_statistic(None, rtrain_jd, rvalid_jd, {"images": [], "annotations": []})
"""
794 816 0 1610
42197 30454 0 72651
[3, 309]
absolute size: 16.958±16.878
relative size: 0.011±0.010
aspect ratio: 0.690±0.422
"""

root_path = 'data/coco/'
# rgb_jd = json.load(open(root_path+'anns/origin/rgb_all.json'))
rtrain_jd = json.load(open(root_path+'annotations/instances_train2017.json'))
rvalid_jd = json.load(open(root_path+'annotations/instances_val2017.json'))
# rtest_jd = json.load(open(root_path+'annotations/origin/rgb_test.json'))

# print_statistic(rgb_jd, rtrain_jd, rvalid_jd, rtest_jd)
print_statistic(None, rtrain_jd, rvalid_jd, {"images": [], "annotations": []})
"""
118287 5000 0 123287
860001 36781 0 896782
[1, 640]
absolute size: 99.496±107.478
relative size: 0.190±0.203
aspect ratio: 1.213±1.337
"""
