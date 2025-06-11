import itertools
import logging
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval
from huicv.evaluation.location_evaluation import LocationEvaluator
from functools import partial

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .cocofmt import CocoFmtDataset



@DATASETS.register_module()
class CocoFmtCityscapeDataset(CocoFmtDataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 data_root=None,
                 corner_kwargs=None,
                 train_ignore_as_bg=True,
                 noise_kwargs=None,
                 merge_after_infer_kwargs=None,
                 min_gt_size=None,
                 **kwargs):
        # add by hui, if there is not corner dataset, create one
        if corner_kwargs is not None:
            assert ann_file[-5:] == '.json', "ann_file must be a json file."
            ann_file = generate_corner_json_file_if_not_exist(ann_file, data_root, corner_kwargs)
            print("load corner dataset json file from {}".format(ann_file))
        if noise_kwargs is not None:
            if 'pseudo_wh' in noise_kwargs and noise_kwargs['pseudo_wh'] is not None:
                ann_file = generate_pesudo_bbox_for_noise_data(ann_file, data_root, noise_kwargs)
            elif 'wh_suffix' in noise_kwargs:
                from .noise_data_utils import get_new_json_file_path
                ann_file, _ = get_new_json_file_path(ann_file, data_root, noise_kwargs['sub_dir'],
                                                     noise_kwargs['wh_suffix'])
            else:
                raise ValueError('one of [pseudo_wh, wh_suffix] must be given')
            print("load noise dataset json file from {}".format(ann_file))

        self.train_ignore_as_bg = train_ignore_as_bg
        self.merge_after_infer_kwargs = merge_after_infer_kwargs

        self.min_gt_size = min_gt_size

        super(CocoFmtDataset, self).__init__(
            ann_file,
            data_root=data_root,
            **kwargs
        )

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if all_iscrowd:
                print('aaa')
            if self.filter_empty_gt and (self.img_ids[i] not in ids_in_cat
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds


