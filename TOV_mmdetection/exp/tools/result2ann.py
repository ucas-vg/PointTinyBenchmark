from pycocotools.coco import COCO
import json
import argparse


def check(coco, res):
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]

                assert ori_ann['id'] == ann['ann_id']
                ori_c, new_c = xywh2centerwh(ori_ann['bbox'])[:2], xywh2centerwh(ann['bbox'])[:2]
                assert round(ori_c[0]) == round(new_c[0]) and round(ori_c[1]) == round(new_c[1])
                for key in ['segmentation', 'area', ]:
                    assert key in ori_ann, ori_ann
                    assert key in ann, ann
                    assert ori_ann[key] == ann[key], f"{key}\n\t{ori_ann}\n\t{ann}"


def load_results(res_jd):
    from collections import defaultdict
    data = defaultdict(dict)
    for res in res_jd:
        data[res['image_id']][res['ann_id']] = res
    return data


def xywh2centerwh(xywh):
    x1, y1, w, h = xywh
    return [x1 + w/2, y1 + h/2, w, h]


def centerwh2xywh(centerwh):
    xc, yc, w, h = centerwh
    return [xc - w/2, yc-h/2, w, h]


def turn_bbox_wh(bbox, new_wh):
    if new_wh[0] > 0 and new_wh[1] > 0:
        x1, y1, w, h = bbox
        xc, yc, w, h = xywh2centerwh([x1, y1, w, h])
        new_bbox = centerwh2xywh([xc, yc, new_wh[0], new_wh[1]])

        cb1, cb2 = xywh2centerwh(new_bbox)[:2], xywh2centerwh(bbox)[:2]
        assert round(cb1[0]) == round(cb2[0]) and round(cb1[1]) == round(cb2[1]), f"{bbox} {cb1} vs {new_bbox} {cb2}"
        bbox = new_bbox
    return bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_ann", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("--det_file", help='such as exp/latest_result.json')
    parser.add_argument("--save_ann", help='such as exp/rr_latest_result.json')
    parser.add_argument("--wh", default=-1, type=int, help="")
    args = parser.parse_args()

    coco = COCO(args.ori_ann)
    res_jd = json.load(open(args.det_file))
    res = coco.loadRes(res_jd)
    imgid2res = load_results(res_jd)

    wh = args.wh
    if isinstance(wh, (int, float)):
        wh = (wh, wh)

    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns_res = res.imgToAnns[im_id]
            anns_raw_res = imgid2res[im_id]
            for ann_res in anns_res:
                ori_ann = coco.loadAnns(ann_res['ann_id'])[0]
                assert ori_ann['id'] == ann_res['ann_id'], f"{ori_ann} vs {ann_res}"

                for key in ['image_id', 'category_id', 'iscrowd']:
                    assert ori_ann[key] == ann_res[key], key

                ori_ann['bbox'] = turn_bbox_wh(ann_res['bbox'], wh)
                for key in ['segmentation', 'area', ]:
                    ori_ann[key] = ann_res[key]

                ann_raw_res = anns_raw_res[ann_res['ann_id']]
                for key in ['geo']:
                    if key in ann_res:
                        ori_ann[key] = ann_res[key]
                    elif key in ann_raw_res:
                        ori_ann[key] = ann_raw_res[key]

    check(coco, res)
    json.dump(coco.dataset, open(args.save_ann, 'w'))
