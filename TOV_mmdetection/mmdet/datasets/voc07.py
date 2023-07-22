import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom_oamil import OAMILCustomDataset

from mmcv.parallel import DataContainer as DC
import torch
from mmdet.core import eval_map, eval_recalls
import itertools
import copy


@DATASETS.register_module()
class OAMILVOCDataset(OAMILCustomDataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        # self.kwargs = {"box_noise_level": box_noise_level}
        # self.kwargs.update(**kwargs)
        super(OAMILVOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

        # set group flag for the sampler | copy from custom.py
        if not self.test_mode:
            self._set_group_flag()

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        # load noisy annotations in training
        if not self.test_mode:
            box_noise_level = self.kwargs['box_noise_level']
            if box_noise_level > 0:
                ann_name = ann_file.split('/')[-1].split('.')[0]
                prefix = self.img_prefix + 'noisy_pkl/'
                ann_file = '{}{}_noise-r{:.1f}.pkl'.format(prefix, ann_name, box_noise_level)
        data_infos = mmcv.load(ann_file)
        return data_infos

    def format_extra(self, data):
        data['img_metas'].data['gt_bboxes'] = data['gt_bboxes'].data

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        return self.data_infos[idx]['ann']

    def get_img_id(self, idx):
        """Get img_id by index. ssd-det

        Args:
            idx (int): Index of data.

        Returns:
            str: Image id of specified index.
        """
        return self.data_infos[idx]['id']

    def results2json(self, results, outfile='./exp/latest_result.json'):
        """Dump the detection results to a COCO style json file. ssd-det

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        if isinstance(results[0], list):
            json_results = []
            for idx in range(len(self)):
                img_id = self.get_img_id(idx)
                result = results[idx]
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = int(img_id)
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = label  # self.cat_ids[label] ssd-det +1
                        if len(bboxes[i]) >= 6:  # add by hui
                            data['ann_id'] = int(bboxes[i][5])
                        json_results.append(data)
            mmcv.dump(json_results, outfile)

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation. ssd-det

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    # Add offset into training data
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        img_info['filename'] = img_info.get('filename', img_info['file_name'])
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        img_info = self.data_infos[idx]
        img_info['filename'] = img_info.get('filename', img_info['file_name'])
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            self.format_extra(data)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 save_result_file=None,  # ssd-det added 22.11.18
                 ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]

        # ssd-det dump annotation for re-train
        save_result_file = './exp/latest_result.json' if save_result_file is None else save_result_file
        self.results2json(results, outfile=save_result_file)

        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
