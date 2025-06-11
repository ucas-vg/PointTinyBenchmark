import itertools
import logging
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from functools import partial

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .cocofmt import CocoFmtDataset


@DATASETS.register_module()
class CocoFmtNoiseDataset(CocoFmtDataset):
    CLASSES = None

    def __init__(self,
                 ann_file,
                 data_root=None,
                 box_noise_level=None,
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
        self.box_noise_level = box_noise_level

        super(CocoFmtDataset, self).__init__(
            ann_file,
            data_root=data_root,
            **kwargs
        )

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        if not self.test_mode:
            self.coco = COCO(ann_file)
            if self.CLASSES is None:
                self.CLASSES = [cat['name'] for cat in self.coco.dataset['categories']]  # add by hui
            # The order of returned `cat_ids` will not
            # change with the order of the CLASSES
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()
            prefix = 'data/coco/noisy_pkl/'
            ann_name = ann_file.split('/')[-1].split('.')[0]
            if self.box_noise_level > 0:
                ann_file = '{}{}_noise-r{:.1f}.pkl'.format(prefix, ann_name, self.box_noise_level)
            else:
                ann_file = '{}{}.pkl'.format(prefix, ann_name, self.box_noise_level)
            data_infos = mmcv.load(ann_file)
            return data_infos
        else:
            self.coco = COCO(ann_file)
            if self.CLASSES is None:
                self.CLASSES = [cat['name'] for cat in self.coco.dataset['categories']]  # add by hui
            # The order of returned `cat_ids` will not
            # change with the order of the CLASSES
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()
            data_infos = []
            total_ann_ids = []
            for i in self.img_ids:
                info = self.coco.load_imgs([i])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
                ann_ids = self.coco.get_ann_ids(img_ids=[i])
                total_ann_ids.extend(ann_ids)
            assert len(set(total_ann_ids)) == len(
                total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
            return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.test_mode:
            img_id = self.data_infos[idx]['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            return self._parse_ann_info(self.data_infos[idx], ann_info)
        else:
            img_id = self.data_infos[idx]['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann_info = self._parse_ann_info(self.data_infos[idx], ann_info)

            ann_info_ = self.data_infos[idx]['ann']
            for i in ann_info_.keys():
                ann_info[i]=ann_info_[i]
            return ann_info
