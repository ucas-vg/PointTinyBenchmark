
import json
import numpy as np


def equal(bbox1, bbox2, eps=1e-6):
    return np.mean(np.abs(np.array(bbox1) - np.array(bbox2))) < eps


def split_annotations(anns):
    weakly_anns = []
    fully_anns = []
    for ann in anns:
        if ann.get("is_fully", False):
            assert equal(ann['bbox'], ann['true_bbox'])
            fully_anns.append(ann)
        else:
            weakly_anns.append(ann)
    return fully_anns, weakly_anns


def semi_annotations(anns, fully_ratio):
    fully_anns, weakly_anns = split_annotations(anns)
    new_fully_num = int(round(len(anns) * fully_ratio) - len(fully_anns))
    assert new_fully_num >= 0
    ann_idxs = list(range(len(weakly_anns)))
    np.random.shuffle(ann_idxs)
    for i in ann_idxs[:new_fully_num]:
        ann = weakly_anns[i]
        ann['bbox'] = ann['true_bbox']
        ann['is_fully'] = 1

    fully_anns, weakly_anns = split_annotations(anns)
    print("fully ratio {}".format(len(fully_anns)/len(anns)))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from huicv.interactivate.path_utils import makedirs_if_not_exist, override_check
    parser = ArgumentParser()
    parser.add_argument('ann', help='ann_file')
    parser.add_argument('save_ann', help='save_ann', default='')
    parser.add_argument('--fully_ratio', help='the ratio of fully supervised annotations', type=float)
    args = parser.parse_args()

    jd = json.load(open(args.ann))
    semi_annotations(jd['annotations'], args.fully_ratio)
    makedirs_if_not_exist(args.save_ann)
    if override_check(args.save_ann):
        json.dump(jd, open(args.save_ann, 'w'), separators=(',', ':'))
        print("save semi annotations in", args.save_ann)
