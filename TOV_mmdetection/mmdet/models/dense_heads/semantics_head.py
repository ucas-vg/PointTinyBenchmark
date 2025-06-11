import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from functools import partial
from mmcv.runner import BaseModule
from inspect import signature
from collections import defaultdict
from mmdet.core.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
import math


def group_by_label(data, labels):
    assert len(labels.shape) == 1
    labels = labels.cpu().numpy().tolist()
    label2data = defaultdict(list)
    for label, d in zip(labels, data):
        label2data[label].append(d)
    return {l: torch.stack(data) for l, data in label2data.items()}


def build_from_type(cfg, **kwargs):
    cls = cfg.pop('type')
    return eval(cls)(**cfg, **kwargs)


def grid_sample(feat, chosen_pts, align_corners):
    """
    # (B=1, num_gt_pts, num_chosen, 2)
    Args:
        feat: shape=(B, C, H, W)
        chosen_pts:  shape=(B, num_gts, num_chosen, 2)
    Returns:
    """
    if align_corners:
        # [0, w-1] -> [-1, 1]
        grid_norm_func = lambda xy, wh: 2 * xy / (wh - 1) - 1
        padding_mode = 'zeros'
    else:
        # [-0.5, w-1+0.5] -> [-1, 1]
        # x -> x' => x' = (2x+1) / w - 1
        grid_norm_func = lambda xy, wh: (2 * xy + 1) / wh - 1  # align_corners=False
        padding_mode = 'border'
    h, w = feat.shape[2:]
    WH = feat.new_tensor([w, h])
    chosen_pts = grid_norm_func(chosen_pts, WH)
    return F.grid_sample(feat, chosen_pts, align_corners=align_corners, padding_mode=padding_mode)


class MultiList:
    """
    Example:
        > ml = MultiList()
        > ml.append(1, 2, 3)
        > ml.append(1, 2, 3)
        > ml.data
        [[1, 1], [2, 2], [3, 3]
    """

    def __init__(self):
        self.data = []

    def append(self, *args):
        if len(self.data) == 0:
            self.data = [[] for arg in args]
        assert len(args) == len(self.data)
        for l, arg in zip(self.data, args):
            l.append(arg)

    def apply(self, fn):
        return [fn(l) for l in self.data]


class PtAndFeat(object):
    def __init__(self, pts=None, valid=None, img_len=None):
        """"""
        self.pts = pts
        self.valid = valid
        self.img_len = img_len

        self.cls_feats = None
        self.ins_feats = None
        self.cls_outs = None
        self.ins_outs = None
        self.cls_prob = None

    def split_each_img(self):
        """
        Returns:
            res: list[PtAndFeat], pts and feats info of each image
        """

        def split_data_each_img(datas, img_lens):
            """
            Args:
                datas: [lvl, (num_all_img, ...)]
                img_lens:
            Returns:
                res: [B, lvl, (num_per_img, ...)]
            """
            res = [[] for lvl in range(len(img_lens))]
            for lvl, img_len in enumerate(img_lens):
                i = 0
                for l in img_len:
                    res[lvl].append(datas[lvl][i:i + l])
                    i += l
            B = len(res[0])
            for r in res:
                assert len(r) == B
            return [[res[lvl][b] for lvl in range(len(res))] for b in range(B)]

        assert self.pts is not None and self.img_len is not None
        res = None
        for key, value in self.__dict__.items():
            if key in ["img_len"]:
                continue
            if value is None:
                continue
            value_list = split_data_each_img(value, self.img_len)
            if res is None:
                res = [PtAndFeat() for _ in value_list]
            for i, v in enumerate(value_list):
                res[i].__setattr__(key, v)
        return res


class PointExtractor(BaseModule):
    """generate pos bag and neg points and extract them features"""

    def __init__(self, strides: tuple, num_classes,
                 pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                 neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3),
                 init_cfg=None):
        super(PointExtractor, self).__init__(init_cfg)
        self.pos_generator = build_from_type(pos_generator, num_classes=num_classes)
        self.neg_generator = build_from_type(neg_generator, num_classes=num_classes)
        self.strides = strides

    def forward(self, cls_feat, ins_feat, gt_r_points, gt_labels, img_metas, gt_points_ignore=None,
                ins_same_as_cls=True):
        """
        Args:
            cls_feat:
            ins_feat:
            gt_r_points:
            img_metas:
            gt_points_ignore:
            ins_same_as_cls:
        Returns:
            pos_data.cls_feats: [num_lvl, (num_gts, num_refine, num_chosen, C)]
            neg_data.cls_feats: [num_lvl, (num_negs, C)]
        """
        if ins_same_as_cls:  # if ins_feat same as cls_feat, extract only one is enough.
            (pos_cls_feat,), pos_data, (neg_cls_feat,), neg_data = self.extract(
                (cls_feat,), gt_r_points, gt_labels, img_metas, gt_points_ignore)
            pos_ins_feat = pos_cls_feat
        else:
            (pos_cls_feat, pos_ins_feat), pos_data, (neg_cls_feat, _), neg_data = self.extract(
                (cls_feat, ins_feat), gt_r_points, gt_labels, img_metas, gt_points_ignore)
        pos_data.cls_feats, pos_data.ins_feats = pos_cls_feat, pos_ins_feat
        neg_data.cls_feats = neg_cls_feat
        return pos_data, neg_data

    def extract(self, all_feats, gt_points_list, gt_labels_list, img_metas_list, gt_points_ignore_list=None):
        """
        Returns:
            pos_feats: [k, num_lvl, (num_gt_all_img, num_refine, num_chosen, C)]
            pos_pts:   [num_lvl, (num_gt_all_img, num_refine, num_chosen, 4)]
            neg_feats: [k, num_lvl, (num_neg_all_img, C)]
            neg_pts:   [num_lvl, (num_neg_all_img, 4)]
        """
        pos_feats, pos_pts, pos_valid = self.pos_generator(self.strides, all_feats, gt_points_list, gt_labels_list,
                                                           img_metas_list, gt_points_ignore_list)
        neg_feats, neg_pts, neg_valid = self.neg_generator(self.strides, all_feats, gt_points_list, gt_labels_list,
                                                           img_metas_list, gt_points_ignore_list)

        pos_img_len = [[len(pts) for im_id, pts in enumerate(pos_pts_lvl)]
                       for lvl, pos_pts_lvl in enumerate(pos_pts)]
        neg_img_len = [[len(pts) for im_id, pts in enumerate(neg_pts_lvl)]
                       for lvl, neg_pts_lvl in enumerate(neg_pts)]
        # cat all img
        for k in range(len(pos_feats)):
            pos_feats[k] = [torch.cat(f) for f in pos_feats[k]]
            neg_feats[k] = [torch.cat(f) for f in neg_feats[k]]
        pos_pts, neg_pts, pos_valid, neg_valid = [[torch.cat(p) for p in data]
                                                  for data in [pos_pts, neg_pts, pos_valid, neg_valid]]
        return pos_feats, PtAndFeat(pos_pts, pos_valid, pos_img_len), \
               neg_feats, PtAndFeat(neg_pts, neg_valid, neg_img_len)


class PtFeatGenerator(object):
    def __init__(self, num_classes, align_corners=False):
        self.align_corners = align_corners
        self.num_classes = num_classes

    def generate(self, *args, **kwargs):
        """generate points and them features for single FPN level of single image """
        raise NotImplementedError()

    def __call__(self, strides, all_feats, gt_points_list, gt_label_list, img_metas_list, gt_points_ignore_list=None):
        """
        Args:
            strides:
            all_feats: [k, num_lvl, (B, C, H, W)]
            gt_points_list: [B, (num_gts, num_refine, 2)]
            gt_label_list: [B, (num_gts, )]
            img_metas_list: (B, )
            gt_points_ignore_list:
        Returns
            feats: [k, num_lvl, B, (..., C)]
            pts: [num_lvl, B, (..., 4)]
            valid: [num_lvl, B, (..., C)]
        """
        assert len(gt_points_list[0].shape) == 3
        assert isinstance(all_feats, (tuple, list))
        k = len(all_feats)
        all_res = MultiList()
        for lvl, stride in enumerate(strides):
            res = MultiList()
            for img_id, gt_points_img in enumerate(gt_points_list):
                img_meta = img_metas_list[img_id]
                gt_labels_img = gt_label_list[img_id]
                feats = [f[lvl][img_id:img_id + 1] for f in all_feats]
                feats, pts, valid = self.generate(feats, img_meta, stride, lvl, gt_points_img, gt_labels_img)
                res.append(*feats, pts, valid)
            all_res.append(*res.data)

        feats, pts, valid = all_res.data[:k], all_res.data[k], all_res.data[k + 1]
        return feats, pts, valid

    def assert_same_size(self, feats):
        h, w = feats[0].shape[-2:]
        for i in range(1, len(feats)):
            ih, iw = feats[i].shape[-2:]
            assert (h, w) == (ih, iw)
        return h, w

    def get_point_valid(self, pts, valid_h, valid_w):
        """
        Args:
            pts: shape=(..., 2)
        Returns:
        """
        valid = torch.zeros(pts.shape[:-1], dtype=torch.bool, device=pts.device)
        valid[(0 <= pts[..., 0]) & (pts[..., 0] < valid_w) & (0 <= pts[..., 1]) & (pts[..., 1] < valid_h)] = 1
        return valid

    def extract_point_feat(self, feat, chosen_pts, stride):
        """
        Args:
            feat: shape=(1, C, H, W)
            chosen_pts: shape=(..., num_chosen, 2)
            stride: float
        Returns:
            point_bag_feats: shape=(..., num_chosen, feat_channel)
        """
        s = chosen_pts.shape[:-2]
        chosen_pts = chosen_pts.flatten(0, -3).unsqueeze(0) / stride  # => (B=1, num_gt_pts, num_chosen, 2)
        # permute(0, 2, 3, 1)[0]: (B=1, feat_c, num_gt_pts, num_chosen) => (num_gt_pts, num_chosen, feat_c)
        bag_feats = grid_sample(feat, chosen_pts, self.align_corners).permute(0, 2, 3, 1)[0]
        # if self.neighbour_cfg.get("nearest_k", -1) > 0:
        #     TestCPRHead.test_extract_point_feat(chosen_pts[:, :-1], feat, bag_feats[:, :-1])
        _, num_chosen, feat_c = bag_feats.shape
        bag_feats = bag_feats.reshape(*s, num_chosen, feat_c)
        return bag_feats

    def append_other_info(self, pts, stride, valid):
        pts_s = torch.full(pts.shape[:-1], stride, dtype=pts.dtype).to(pts.device)  # add fpn stride
        pts = torch.cat([pts, pts_s.unsqueeze(-1)], dim=-1)  #
        return pts


class AnchorPtFeatGenerator(PtFeatGenerator):
    def __init__(self, scale_factor=None, **kwargs):
        """
        Args:
            scale_factor: rescale of the feature when sample point from feature map
        """
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def generate(self, feats, img_meta, stride, fpn_lvl, centers, labels):
        """
        Args:
            feats: list[Tensor], [k, (1, C, H, W)]
            img_meta: dict
            stride:
            fpn_lvl:
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            pt_feats: list[Tensor], [k, (H, W, C)]
            pts: (H, W, 3), 3 is (x, y, lvl)
            valid: (H, W, 1)
        """
        if self.scale_factor and self.scale_factor != 1.0:
            feats = [F.interpolate(feat, self.scale_factor, mode='bilinear', align_corners=self.align_corners,
                                   recompute_scale_factor=False) for feat in feats]
        h, w = self.assert_same_size(feats)
        pts, valid = self.anchor_points(h, w, *img_meta['pad_shape'][:2], stride, feats[0].device)
        pts = self.append_other_info(pts, stride, valid)
        pt_feats = [feat.permute(0, 2, 3, 1).squeeze(0) for feat in feats]
        # valid_class = pts.new_zeros(*valid.shape, self.num_classes)  # repeat valid to num_class
        # valid_class[..., :] = valid[..., None]
        return pt_feats, pts, valid[..., None]

    def anchor_points(self, h, w, valid_h, valid_w, stride, device):
        x, y = torch.arange(w).to(device), torch.arange(h).to(device)
        y, x = torch.meshgrid(y, x)
        pts = torch.stack([x, y], dim=-1) * stride + stride / 2
        return pts, self.get_point_valid(pts, valid_h, valid_w)


class OutCirclePtFeatGenerator(AnchorPtFeatGenerator):
    def __init__(self, radius, class_wise=False, keep_wh=False, **kwargs):
        self.radius = radius
        self.class_wise = class_wise
        self.keep_wh = keep_wh
        super().__init__(**kwargs)

    def generate(self, feats, img_meta, stride, fpn_lvl, centers, labels):
        """
        Args:
            feats: list[Tensor], (k, (1, C, H, W))
            img_meta: dict
            centers: (num_gts, num_refine, 2)
            labels: (num_gts, )
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            pt_feats: list[Tensor], [k, (num, C)]
            pts: (num, 3), 3 is (x, y, lvl)
            valid: (num, num_class)
        """
        pt_feats, pts, valid = super().generate(feats, img_meta, stride, fpn_lvl, centers, labels)
        h, w, _ = valid.shape
        pts, valid = pts.flatten(0, -2), valid.flatten(0, -2)

        repeat_l = [1] * len(valid.shape[:-1]) + [self.num_classes]
        valid = valid.repeat(*repeat_l)
        if self.class_wise:
            label2centers = group_by_label(centers, labels)
            for label in label2centers:
                centers = label2centers[label].flatten(0, 1)  # (num_gt * num_refine, 2)
                dist = torch.cdist(pts[..., :2], centers, p=2)
                chosen = dist.min(dim=1)[0] >= stride * self.radius
                valid[..., label] = valid[..., label].float() * chosen.float()
        else:
            centers = centers.flatten(0, 1)  # (num_gt * num_refine, 2)
            dist = torch.cdist(pts[..., :2], centers, 2)
            chosen = dist.min(dim=1)[0] >= stride * self.radius
            valid = valid.float() * chosen[..., None].float()
        if self.keep_wh:
            valid = valid.reshape(h, w, -1)
            pts = pts.reshape(h, w, -1)
        else:
            pt_feats = [feat.flatten(0, -2) for feat in pt_feats]
        return pt_feats, pts, valid.bool()


class PointRefiner(object):
    def __init__(self, strides, gt_alpha=0.5, merge_th=0.05, refine_th=0.05,
                 classify_filter=False, refine_pts_extractor=None, return_score_type='mean',
                 debug=False):
        self.gt_alpha = gt_alpha
        self.merge_th = merge_th
        self.refine_th = refine_th
        self.strides = strides
        self.use_classify_filter = classify_filter

        self.refine_pts_extractor = refine_pts_extractor
        # self.pos_generator = build_from_type(pos_generator)
        # self.other_generators = [build_from_type(g) for g in other_generators]
        self.return_score_type = return_score_type
        self.debug = debug

    def grid_merge_per_class(self, grid_cls_prob, dist, gt_prob, gt_r_points, num_refine):
        valid = (grid_cls_prob > self.merge_th) & (grid_cls_prob > gt_prob * self.gt_alpha)
        _, closest_gt_idx = dist[valid].min(dim=1)
        chosen_pts = []
        for idx in closest_gt_idx:
            gt_idx = idx % num_refine

    # def grid_merge(self, grid_cls_prob, grid_pts, gt_r_pts, gt_labels, num_refine):
    #     """
    #     1. assign grid point to each object
    #     Returns:
    #     """
    #     grid_pts, grid_cls_prob = grid_pts.flatten(0, -2), grid_cls_prob.flatten(0, -2)
    #     dist = torch.cdist(grid_pts[..., :2], gt_r_pts[..., :2], p=2)
    #
    #     cls2gt_idx = defaultdict(list)
    #     for i, l in enumerate(gt_labels):
    #         cls2gt_idx[l].append(i)
    #     for l, gt_idx in cls2gt_idx.items():
    #         gt_r_pts_l = gt_r_pts[gt_idx]
    #         dist_l = dist[:, gt_idx]
    #         grid_cls_prob_l = grid_cls_prob[:, l]

    def nearest_filter(self, bag_pts, gt_r_pts, gt_labels, class_wise=True):
        """
        Args:
            bag_pts: shape=(num_gts, num_refine, num_chosen, 3)
            gt_r_pts:  (num_gts, num_refine, 2)
            gt_labels: (num_gts,)
            class_wise:
        Returns:
        """

        def filter(bag_pts, gt_r_pts):
            num_gts, num_refine, num_chosen, _ = bag_pts.shape
            dist = torch.cdist(bag_pts.flatten(0, -2)[..., :2], gt_r_pts.flatten(0, -2)[..., :2], p=2)
            _, closest_gt_idx = dist.min(dim=1)
            closest_gt_idx = closest_gt_idx.reshape(num_gts * num_refine, num_chosen)
            cur_gt_idx = torch.arange(len(closest_gt_idx)).reshape(-1, 1).to(closest_gt_idx.device)
            close_valid = (closest_gt_idx == cur_gt_idx).reshape(num_gts, num_refine * num_chosen)
            return close_valid

        if class_wise:
            gt_idx = torch.arange(len(bag_pts))
            label2gt_r_pts = group_by_label(gt_r_pts, gt_labels)
            label2bag_pts = group_by_label(bag_pts, gt_labels)
            label2gt_idx = group_by_label(gt_idx, gt_labels)

            num_gts, num_refine, num_chosen, _ = bag_pts.shape
            valid = gt_labels.new_ones((num_gts, num_refine * num_chosen), dtype=torch.bool)
            for l in label2gt_r_pts:
                gt_r_pts, bag_pts, gt_idx = label2gt_r_pts[l], label2bag_pts[l], label2gt_idx[l]
                if len(gt_r_pts) > 1:
                    valid[gt_idx] = filter(bag_pts, gt_r_pts)
            return valid
        else:
            return filter(bag_pts, gt_r_pts)

    def classify_filter(self, bag_cls_prob, gt_labels):
        """
        Args:
            bag_cls_prob:  (num_gts, ..., num_class)
            gt_labels: (num_gts, )
        Returns:
        """
        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        _, classify_res = bag_cls_prob.max(dim=-1)
        shape = [len(classify_res)] + [1] * (len(classify_res.shape) - 1)
        valid = classify_res == gt_labels.reshape(*shape)
        return valid.reshape(num_gts, num_refine * num_chosen)

    def graph_filter(self, strides, bag_valid):
        """
        Args:
            strides: (..., 1)
            bag_valid: (num_gts, num_refine, num_chosen)
        Returns:
        """
        stride = strides.reshape(-1, 1)[0].tolist()[0]
        assert (strides == stride).all(), ""
        gt_valid = bag_valid[..., -1].clone()
        bag_valid[..., -1] = True
        self.refine_pts_extractor.pos_generator.filter_depend(stride, 2.0, bag_valid)
        bag_valid[..., -1] = gt_valid
        return bag_valid.flatten(0, 1)

    def inside_img(self, bag_pts, img_shape):
        num_gts, num_refine, num_chosen, _ = bag_pts.shape
        bag_pts = bag_pts.reshape(num_gts, num_refine * num_chosen, -1)
        h, w, _ = img_shape
        x, y = bag_pts[..., 0], bag_pts[..., 1]
        return (x < w) & (x >= 0) & (y < h) & (y >= 0)

    def refine_single(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_r_points, gt_labels,
                      img_meta, gt_true_bboxes, not_refine=None):
        """
        refine point in single image
        Args:
            bag_data.cls_prob: [lvl, (num_gts, num_refine, num_chosen, num_class)]
            bag_data.valid: [lvl, (num_gts, num_refine, num_chosen, 1)]
            grid_data.cls_prob: [lvl, (num_negs, num_class)]
            grid_data.valid: [lvl, (num_negs, num_class)]
            gt_labels: (num_gts,)
            gt_r_points: (num_gts, num_refine, 2)
        Returns:
        """
        bag_cls_prob, bag_ins_outs, grid_cls_prob = bag_data.cls_prob, bag_data.ins_outs, grid_data.cls_prob
        bag_pts, bag_valid = bag_data.pts, bag_data.valid
        grid_pts, grid_valid = grid_data.pts, grid_data.valid

        # 1. fpn lvl
        assert len(bag_cls_prob) == 1 == len(grid_cls_prob) == len(grid_pts) == len(bag_pts)
        bag_cls_prob, bag_ins_outs, bag_pts = bag_cls_prob[0], bag_ins_outs[0], bag_pts[0]
        grid_cls_prob, grid_pts = grid_cls_prob[0], grid_pts[0]
        bag_valid, grid_valid = bag_valid[0], grid_valid[0]
        stride = self.strides[0]
        assert grid_cls_prob.shape[-1] == bag_cls_prob.shape[-1]

        # 2. split gt prob out
        gt_cls_prob = bag_cls_prob[..., 0, :]
        gt_r_pts = bag_pts[..., 0, :]
        # ubaozheng gt point
        # assert (gt_r_pts[:, :, 0, :2] == gt_r_points).all(), f"{gt_r_pts.shape} vs {gt_r_points.shape}"

        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        gt_idx = torch.arange(len(gt_labels))
        merge_valid = bag_valid.reshape(num_gts, num_refine * num_chosen).bool()
        # 3. assign point
        merge_valid &= self.nearest_filter(bag_pts, gt_r_pts, gt_labels)
        if self.use_classify_filter:
            merge_valid &= self.classify_filter(bag_cls_prob, gt_labels)
        # 4. prob > th & prob > gt_prob * a
        bag_cls_prob = bag_cls_prob[gt_idx, ..., gt_labels].reshape(num_gts, num_refine * num_chosen)
        gt_cls_prob = gt_cls_prob[gt_idx, 0, ..., gt_labels].reshape(num_gts, 1)
        # gt_cls_prob = gt_cls_prob.repeat(1, 1, num_chosen).flatten(1)
        merge_valid &= (bag_cls_prob > self.merge_th) & (bag_cls_prob > gt_cls_prob * self.gt_alpha)
        # graph filter
        # merge_valid &= self.graph_filter(bag_pts[..., -1:], merge_valid.reshape(num_gts, num_refine, num_chosen))
        # inside image filter
        merge_valid &= self.inside_img(bag_pts, img_meta['img_shape'])

        # 5. merge
        bag_pts = bag_pts.reshape(num_gts, num_refine * num_chosen, 3)
        bag_cls_prob = bag_cls_prob * merge_valid.float()
        bag_pts_weight = bag_cls_prob / (bag_cls_prob.sum(dim=1, keepdim=True) + 1e-8)
        refine_pts = (bag_pts[..., :2] * bag_pts_weight.unsqueeze(dim=-1)).sum(dim=1)  # (num_gts, 2)

        refine_scores = bag_cls_prob.sum(dim=-1) / ((bag_cls_prob > 0).float().sum(dim=-1) + 1e-8)  # (num_gts, )
        cur_not_refine = (refine_scores < self.refine_th)
        not_refine = cur_not_refine if not_refine is None else not_refine | cur_not_refine
        refine_pts[not_refine] = gt_r_points[:, 0][not_refine]

        if self.return_score_type == 'max':
            refine_scores = bag_cls_prob.max(dim=-1)[0]
            refine_scores[refine_scores == 0] = self.refine_th / 2
        elif self.return_score_type == 'mean':
            pass
        else:
            raise ValueError

        chosen_pts = [bag_pts[gt_i][chosen][:, :2] for gt_i, chosen in enumerate(bag_pts_weight > 0)]

        return refine_pts, refine_scores, not_refine, chosen_pts, merge_valid

    def __call__(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_r_points, gt_labels, img_metas,
                 gt_true_bboxes, not_refine=None):
        """
        1.
        Args:
            bag_data.cls_prob: [num_lvl, (num_gts_all_img, num_refine, num_chosen, C)]
            grid_data.cls_prob: [num_lvl, (num_negs_all_img, C)]
            gt_labels: [B, (num_gt, )]
        Returns:
        """
        # split data by each img => [B, num_lvl, (num_gts, num_refine, num_chosen, C)]
        bag_data_list, grid_data_list = bag_data.split_each_img(), grid_data.split_each_img()

        if gt_true_bboxes is None:
            gt_true_bboxes = [None] * len(gt_labels)
        if not_refine is None:
            not_refine = [None] * len(gt_labels)

        refine_pts, refine_scores, not_refine, chosen_pts, merge_valid = multi_apply(
            self.refine_single, bag_data_list, grid_data_list, gt_r_points, gt_labels, img_metas, gt_true_bboxes,
            not_refine)
        refine_pts_final = refine_pts
        not_refine_final = not_refine
        if self.debug:
            for i, img_meta in enumerate(img_metas):
                refine_pts_d, chosen_pts_d, gt_r_points_d, gt_labels_d, img_metas_d, gt_true_bboxes_d, not_refine_d, refine_scores_d = \
                    refine_pts[i:i + 1], chosen_pts[i:i + 1], gt_r_points[i:i + 1], gt_labels[i:i + 1], img_metas[
                                                                                                        i:i + 1], \
                    gt_true_bboxes[i:i + 1], not_refine[i:i + 1], refine_scores[i:i + 1]
                TestCPRHead.get(self).test_refine_point(refine_pts_d, None, chosen_pts_d,
                                                        gt_r_points_d, gt_labels_d, img_metas_d, gt_true_bboxes_d,
                                                        not_refine_d, refine_scores_d)
                # TestCPRHead.get(self).test_grid(grid_data_list, gt_labels, img_metas)
        return refine_pts, refine_scores, not_refine, merge_valid


@HEADS.register_module()
class SemanticsHead(AnchorFreeHead):
    """P2PNet head.
    arxiv:2107.12746
    Args:
    """

    def __init__(self, num_classes, in_channels,
                 # loss_cls=dict(
                 #     type='FocalLoss',
                 #     use_sigmoid=True,
                 #     gamma=2.0,
                 #     alpha=0.25,
                 #     loss_weight=1.0
                 # ),
                 debug=False,
                 num_cls_fcs=0,
                 point_anchor=[(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)],  # Grid
                 # point_anchor=[(0, 0), (0, 0), (0, 0), (0, 0)],  # Center
                 assign_before_pred=False,
                 pts_gamma=100. / 8,  # γ to 100., 8 is stride
                 reg_norm=1. / 8,  # 8 is stride
                 sample_r=7,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
                 loss_mil=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25),
                 loss_type=0,
                 loss_cfg=dict(
                     with_neg=True,
                     neg_loss_weight=0.75,
                     refine_bag_policy='only_refine_bag',
                     random_remove_rate=0.4,
                     with_gt_loss=True,
                     gt_loss_weight=0.125,
                     with_mil_loss=True,
                     with_pts_reg=True,
                 ),
                 normal_cfg=dict(
                     prob_cls_type='sigmoid',
                     out_bg_cls=False
                 ),
                 point_refiner=dict(
                     merge_th=0.1,
                     refine_th=0.1,
                     classify_filter=True,
                 ),
                 loss_reg=dict(
                     type='SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=0.5),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_out',  # ?必须与成员变量对应?
                         std=0.01,
                         bias_prob=0.01
                     )
                 ),
                 **kwargs):

        self.point_anchor = torch.FloatTensor(point_anchor)
        self.num_points = len(point_anchor)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:  # self.cls_out_channels
            self.num_cls_out = num_classes  # sigmoid（多个二分类，相加!=1）
        else:
            self.num_cls_out = num_classes + 1  # softmax（多分类，加背景类，相加==1）
        self.loss_cls_type = loss_cls['type']
        self.binary_ins = loss_mil['binary_ins']
        self.ins_share_head_feat = False
        self.ins_share_head_classifier = False
        self.num_cls_fcs = num_cls_fcs
        self.loss_type = loss_type
        self.loss_cfg = loss_cfg
        self.debug = debug
        self.sample_r = sample_r
        # self.refine_pts_extractor = PointExtractor(**refine_pts_extractor, strides=[8], num_classes=num_classes)
        self.point_refiner = PointRefiner(**point_refiner, debug=self.debug, strides=[8],
                                          refine_pts_extractor=None, )
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, init_cfg=init_cfg, **kwargs)

        self.point_generators = [PointGenerator() for _ in self.strides]

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler)

        self.assign_before_pred = assign_before_pred
        self.pts_gamma = pts_gamma
        self.reg_norm = reg_norm
        self.normal_cfg = normal_cfg
        self.loss_reg = build_loss(loss_reg)  # 不用super().loss_bbox
        self.loss_mil = build_loss(loss_mil)
        self.pos_threshold = point_refiner['merge_th']
        TestCPRHead.DO_TEST = debug

    def _init_layers(self):
        """ build head architecture
        Returns:
        """
        self.cls_convs = nn.ModuleList()

        self.ins_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias)
            )
            if not self.ins_share_head_feat:
                self.ins_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias)
            )

        # self.cls_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, 1, 1, 0)
        # self.cls_out = nn.Conv2d(self.point_feat_channels, self.num_cls_out, 1, 1, 0)

        self.cls_fcs = nn.ModuleList()
        self.ins_fcs = nn.ModuleList()
        for i in range(self.num_cls_fcs):
            self.cls_fcs.append(nn.Linear(chn, self.fc_out_channels))
            if not self.ins_share_head_feat:
                self.ins_fcs.append(nn.Linear(chn, self.fc_out_channels))
            chn = self.fc_out_channels

        self.cls_out = nn.Linear(chn, self.num_cls_out)
        if not self.ins_share_head_classifier:
            num_ins_out = self.num_cls_out * 2 if self.binary_ins else self.num_cls_out
            self.ins_out = nn.Linear(chn, num_ins_out)
        else:
            assert not self.binary_ins
            self.ins_out = self.cls_out
        # # num_cls_out*k
        self.reg_out = nn.Conv2d(self.feat_channels,
                                 len(self.point_anchor) * 2, 3, padding=1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      pseudo_boxes,
                      dynamic_weight,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_true_bboxes=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        assert len(self.strides) == 1
        # if self.strides[0] == 4:
        x = [x[int(self.strides[0] / 4 - 1)]]
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, pseudo_boxes=pseudo_boxes, dynamic_weight=dynamic_weight,
                           gt_bboxes_ignore=gt_bboxes_ignore,
                           gt_true_bboxes=gt_true_bboxes)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        """forward the feats of backbone output to the head.

        Args:
        Returns:
        """
        cls_feats, ins_feat, pts_outs = multi_apply(self.forward_single, feats)
        return cls_feats, ins_feat, pts_outs

    def random_remove(self, *all_pts, random_remove_rate=0):
        """
        all_pts: [k, (..., a)]
        Returns:
        """
        if random_remove_rate > 0:
            for pts in all_pts:
                valid = pts[..., -1]
                remove = torch.rand(valid.shape) < random_remove_rate
                valid[remove] = 0.
                pts[..., -1] = valid

    def get_pts_outs(self, pts_cls_feats, pts_ins_feats=None):
        """
        Args:
            pts_cls_feats: [num_lvl, (..., C)]
            pts_ins_feats: [num_lvl, (..., C]
        Returns:
            cls_outs: [num_lvl, (..., num_class)]
            ins_outs: [num_lvl, (..., num_class)]
        """

        def forward_with_fc(feat, fcs):
            feat = feat.flatten(0, -2)
            for i, fc in enumerate(fcs):
                feat = self.relu(fc(feat))
            return feat

        def get_outs_single(cls_f, ins_f=None):
            shape = cls_f.shape
            cls_f = forward_with_fc(cls_f, self.cls_fcs)
            cls_o = self.cls_out(cls_f).reshape(*shape[:-1], -1)

            if ins_f is None:
                return cls_o,
            ins_f = forward_with_fc(ins_f, self.ins_fcs) if not self.ins_share_head_feat else cls_f
            ins_o = self.ins_out(ins_f).reshape(*shape[:-1], -1) if not \
                (self.ins_share_head_feat and self.ins_share_head_classifier) else cls_o
            return cls_o, ins_o

        if pts_ins_feats is None:
            cls_outs, = get_outs_single(pts_cls_feats)
            return cls_outs
        else:
            cls_outs, ins_outs = get_outs_single(pts_cls_feats, pts_ins_feats)
            return cls_outs, ins_outs

    def forward_single(self, feat):
        cls_feat = feat
        pts_feat = feat

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        if not self.ins_share_head_feat:
            ins_feat = cls_feat
            for ins_conv in self.ins_convs:
                ins_feat = ins_conv(ins_feat)
        else:
            ins_feat = cls_feat

        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        #
        # cls_out = self.cls_out(cls_feat)
        pts_out = self.reg_out(pts_feat)
        return cls_feat, ins_feat, pts_out

    def get_pred_points(self, cls_feats, ins_feats, pts_outs, img_metas):
        """
        1. pred_pts = anchor_pts + pred_pts * stride * gama

        Args:
            cls_outs: [num_lvl, (B, K*C, H, W)]
            pts_outs: [num_lvl, (B, K*2, H, W)]
            img_metas:
        Returns:
            pred_pts: (B, num_pts_all_lvl*k, 2)
            valid_flag: (B, lvl*w*h*k)
            cls_outs: (B, lvl*w*h*k, num_cls_out)
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_feats]
        assert len(featmap_sizes) == len(self.point_generators)  # num_lvl
        # ----------------------------------------------------
        # cls_outs, pts_out >>>   (B, lvl*w*h, k*num_cls_out)     >>> (B, lvl*w*h, k, num_cls_out)
        #                         (B, lvl*w*h, k*2)                   (B, lvl*w*h, k, 2)
        cls_feats = (torch.cat(
            [cls_feat.reshape(cls_feat.size(0), cls_feat.size(1), -1) for cls_feat in cls_feats], -1)).permute(0, 2, 1)
        cls_feats = cls_feats.reshape(*cls_feats.shape[:2], self.num_points, cls_feats.shape[-1])
        ins_feats = (torch.cat(
            [ins_feat.reshape(ins_feat.size(0), ins_feat.size(1), -1) for ins_feat in ins_feats], -1)).permute(0, 2, 1)
        ins_feats = ins_feats.reshape(*ins_feats.shape[:2], self.num_points, ins_feats.shape[-1])

        pts_outs = (torch.cat(
            [pts_out.reshape(pts_out.size(0), pts_out.size(1), -1) for pts_out in pts_outs], -1)).permute(0, 2, 1)
        pts_outs = pts_outs.reshape(*pts_outs.shape[:2], self.num_points, 2)

        # 1.1 get reference points
        device = cls_feats[0].device
        center_list, valid_flag = self.get_points(featmap_sizes, img_metas, device)
        # [B, lvl, (w*h, 3)],[B, lvl, (w*h)]

        center_list = torch.stack([torch.cat(i) for i in center_list])  # (B, lvl*w*h, 3)
        valid_flag = torch.stack([torch.cat(i) for i in valid_flag])  # (B, lvl*w*h)
        anchor_pts = (center_list.unsqueeze(2)).repeat(1, 1, self.num_points, 1)  # (B, lvl*w*h, k, 3)
        anchor_pts[..., :2] += self.point_anchor.to(device) * anchor_pts[..., -1:]
        valid_flag = (valid_flag.unsqueeze(2)).repeat(1, 1, self.num_points)  # (B, lvl*w*h, k)

        # 1.2 pred_pts = anchor_pts + pred_pts * stride * gama
        # (B, lvl*w*h, k, num_cls_out)
        # pred_pts = anchor_pts[..., :2] + pts_outs * anchor_pts[..., -1:] * self.pts_gamma  # (4,320,4,2)
        pred_pts = anchor_pts[..., :2] + pts_outs * self.pts_gamma * anchor_pts[..., -1:]  # (4,320,4,2)
        pred_pts = torch.cat([pred_pts, anchor_pts[..., -1:]], dim=-1)
        pred_pts = pred_pts.reshape(pred_pts.shape[0], -1, 3)  # (B, num_pts_all_lvl*k, 2)
        cls_feats = cls_feats.reshape(cls_feats.size(0), -1, cls_feats.shape[-1])
        ins_feats = ins_feats.reshape(ins_feats.size(0), -1, ins_feats.shape[-1])  # (B, lvl*w*h*k, num_cls_out)
        valid_flag = valid_flag.reshape(valid_flag.size(0), -1)  # (B, lvl*w*h*k)
        anchor_pts = anchor_pts.reshape(anchor_pts.shape[0], -1, 3)
        return anchor_pts, pred_pts, valid_flag, cls_feats, ins_feats

    def get_cls_prob(self, cls_out):
        """
        Args:
            cls_out: (..., C*K),
        Returns:
        """
        prob_cls_type = self.normal_cfg["prob_cls_type"]
        shape = cls_out.shape[:-1]
        cls_out = cls_out.reshape(*shape, self.num_cls_out, -1)
        if prob_cls_type == 'softmax':
            prob_cls = cls_out.softmax(dim=-2)
        elif prob_cls_type == 'sigmoid':
            prob_cls = cls_out.sigmoid()
        elif prob_cls_type == 'normed_sigmoid':
            prob_cls = cls_out.sigmoid()
            p = self.normal_cfg.get("normed_sigmoid_p", 1)
            prob_cls = F.normalize(prob_cls, p=p, dim=-2)
        else:
            raise ValueError()
        return prob_cls.reshape(*shape, -1)

    def loss(self, cls_feats, ins_feats, pts_outs, gt_bboxes, gt_labels, img_metas, pseudo_boxes, dynamic_weight,
             gt_bboxes_ignore=None,
             gt_true_bboxes=None,
             gt_weights=None):
        """ calculate loss with forward output and ground truth
                Args:
                    cls_outs: [num_level, (B, k*num_cls_out, H, W)]
                    pts_outs: [num_level, (B, len(self.point_anchor)*2, H, W)]
                    gt_bboxes: [B, (num_gt, 4)]
                    gt_labels: [B, (num_gt)]
                    img_metas: [dict]
                    gt_bboxes_ignore:
                Returns:
         """
        for gb in gt_bboxes:
            assert len(gb) > 0, gt_bboxes
        feat_map_shape = cls_feats[0].shape[-2:]
        anchor_pts, pred_pts, valid_flag, cls_feats, ins_feats = self.get_pred_points(cls_feats, ins_feats, pts_outs,
                                                                                      img_metas)
        # assign and sample
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)  # [B, (num_gt, 2)]
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore)  # [B, (num_gt_ig, 4)]
        # proposal = anchor_pts if self.assign_before_pred else pred_pts
        anchor_pts = [i for i in anchor_pts]
        pred_pts = [i for i in pred_pts]
        valid_flag_list = [i for i in valid_flag]
        cls_feats_list = [i for i in cls_feats]
        ins_feats_list = [i for i in ins_feats]

        pos_data, neg_data = self.get_targets(
            anchor_pts,  # (B, lvl*w*h*k, 2)#anchor_pts,  # (B, lvl*w*h, k, 3) >>>
            pred_pts,
            valid_flag_list,
            cls_feats_list,
            ins_feats_list,
            pseudo_boxes,
            gt_points, gt_labels, img_metas, gt_points_ignore
        )  # [B, (lvl*w*h*k)],[B, (lvl*w*h*k, 2)]

        gt_r_points = [pts.reshape(len(labels), -1, *pts.shape[1:]) for pts, labels in zip(gt_points, gt_labels)]
        #
        pos_data.cls_prob = [self.get_cls_prob(bag_cls_feat_lvl) for bag_cls_feat_lvl in pos_data.cls_outs]
        neg_data.cls_prob = [self.get_cls_prob(grid_cls_outs_lvl) for grid_cls_outs_lvl in neg_data.cls_outs]

        refined_points, refined_scores, not_refine, merge_valid = self.point_refiner(pos_data, neg_data, gt_r_points,
                                                                                     gt_labels,
                                                                                     img_metas,
                                                                                     gt_true_bboxes, not_refine=None)

        gt_labels_all = torch.cat(gt_labels, dim=0)
        gt_weights = [torch.FloatTensor([1.0] * len(l)) for l in gt_labels] if gt_weights is None else gt_weights
        gt_weights = torch.cat(gt_weights, dim=0).to(gt_labels_all.device)

        pseudo_pts = [pts.unsqueeze(1).detach() for pts in refined_points]

        self.show_imgs(neg_data.cls_outs[0], img_metas, gt_labels, pos_data.img_len[0], pseudo_boxes, gt_true_bboxes,
                       feat_map_shape)

        return getattr(self, f'loss{self.loss_type}')(pos_data, neg_data, gt_labels_all, gt_r_points, pseudo_pts,
                                                      merge_valid,
                                                      dynamic_weight,
                                                      img_metas,
                                                      gt_true_bboxes,
                                                      gt_weights, )

        return loss_dict_all

    def show_imgs(self, cls_scores, img_metas, gt_labels, img_len, pseudo_boxes, gt_bboxes, feat_map_shape):
        import cv2
        # cls_scores = F.normalize(cls_scores, p=1, dim=-2)
        cls_scores = cls_scores.reshape(len(img_metas), -1, cls_scores.shape[-1])
        # gt_labels = gt_labels.split(img_len)
        for i in range(len(img_metas)):
            gt_box = gt_bboxes[i]
            pos_box = pseudo_boxes[i]

            img_meta = img_metas[i]
            h, w, = feat_map_shape
            cls_scores_ = cls_scores[i]
            cls_scores_ = cls_scores_.reshape(h, w, cls_scores.shape[-1])
            cls_scores_ = cls_scores_.unsqueeze(0).expand(len(gt_labels[i]), *cls_scores_.shape)[
                torch.arange(len(gt_labels[i])), ..., gt_labels[i]]

            pos_box = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
            gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
            ims = cv2.imread(img_meta['filename'])
            im_h, im_w, _ = img_meta['img_shape']
            ims = cv2.resize(ims, (im_w, im_h))
            for k in range(len(gt_labels[i])):
                ims = cv2.rectangle(ims, (gt_box[k, 0], gt_box[k, 1]), (gt_box[k, 2], gt_box[k, 3]),
                                    color=(0, 255, 0))
            # for i in range(len(gt_labels[i])):
            #     ims = cv2.rectangle(ims, (pos_box[i, 0], pos_box[i, 1]), (pos_box[i, 2], pos_box[i, 3]),
            #                          color=(0, 255, 0))

            for j in range(len(gt_labels[i])):
                heatmap = cls_scores_[j]
                # heatmap = heatmap.permute(1, 0)
                # heatmap[heatmap>0.1]=1
                heatmap = np.array(heatmap.cpu().detach())
                heatmapshow = None
                heatmap[0, 0] = 1
                heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)

                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                pad_h, pad_w, _ = img_meta['pad_shape']
                im_h, im_w, _ = img_meta['img_shape']
                heatmapshow = cv2.resize(heatmapshow, (w * self.strides[0], h * self.strides[0]),
                                         interpolation=cv2.INTER_NEAREST)
                heatmapshow = heatmapshow[:im_h, :im_w, :]
                img = cv2.imread(img_meta['filename'])
                img = cv2.resize(img, (im_w, im_h))
                img = cv2.rectangle(img, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                    color=(0, 255, 0))
                img = cv2.rectangle(img, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                    color=(0, 255, 255))
                heatmapshow = cv2.rectangle(heatmapshow, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                            color=(0, 255, 0))
                heatmapshow = cv2.rectangle(heatmapshow, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                            color=(0, 255, 255))

                cv2.namedWindow("ims1", 0)
                cv2.resizeWindow("ims1", 640, 480)
                cv2.imshow('ims1', ims)
                cv2.namedWindow("ht", 0)
                cv2.resizeWindow("ht", im_w * 4, im_h * 4)
                cv2.imshow('ht', heatmapshow)
                cv2.namedWindow("im", 0)
                cv2.resizeWindow("im", im_w * 4, im_h * 4)
                cv2.imshow('im', img)
                cv2.waitKey()
                cv2.destroyAllWindows()

    def loss0(self, pos_data: PtAndFeat, neg_data: PtAndFeat, gt_labels_all, gt_r_points, pseudo_pts, merge_valid,
              dynamic_weight,
              img_metas, gt_true_bboxes=None, gt_weights=None):
        """
        Args:
            pos_data.cls_outs: [num_lvl, (num_gts, num_refine, num_chosen, num_class)], 2 is cls and ins
            pos_data.ins_outs: [num_lvl, (num_gts, num_refine, num_chosen, num_class)]
            pos_data.pts: [num_lvl, (num_gts, num_refine, num_chosen, 3)]
            pos_data.valid: [num_lvl, (num_gts, num_refine, num_chosen, num_class)]
            neg_data.cls_outs: [num_lvl, (num_negs, num_class)]
            neg_data.pts:  [num_lvl, (num_negs, 3)]
            neg_data.valid: [num_lvl, (num_negs, num_class)]
            gt_labels_all: (num_gts,)
            gt_r_points: [B, (num_gt_per_img, num_refine, 2)]
            pseudo_pts: [B, (num_gt_per_img, num_refine, 2)]
            gt_true_bboxes:
            gt_weights: (num_gts, )
        Returns:
        """

        from mmdet.models.losses.utils import weight_reduce_loss
        pos_cls_outs, pos_ins_outs, neg_cls_outs = pos_data.cls_outs, pos_data.ins_outs, neg_data.cls_outs
        pos_pts, neg_pts = pos_data.pts, neg_data.pts
        pos_valid, neg_valid = pos_data.valid, neg_data.valid
        assert len(pos_cls_outs) == len(neg_cls_outs) == 1, f"{len(pos_cls_outs)}, {len(neg_cls_outs)}"
        pos_cls_outs, pos_ins_outs, pos_pts = pos_cls_outs[0], pos_ins_outs[0], pos_pts[0]
        neg_cls_outs, neg_pts = neg_cls_outs[0], neg_pts[0]
        pos_valid, neg_valid = pos_valid[0], neg_valid[0]

        losses = {}
        num_gts, num_refine, num_chosen, _ = pos_pts.shape
        if self.loss_cfg.get('with_gt_loss', False):
            gt_cls_outs = pos_cls_outs[..., 0, :].reshape(num_gts * num_refine, -1)
            gt_cls_prob = self.get_cls_prob(gt_cls_outs)
            gt_loss_type = self.loss_cfg.get('gt_loss_type', 'gt_refine')
            if gt_loss_type == 'mil':
                raise NotImplementedError
            else:
                if gt_loss_type == 'gt_refine':
                    gt_labels_rep = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_valid = pos_valid[..., 0, :].reshape(num_gts * num_refine, -1)
                    gt_weights_rep = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_weights_rep = gt_valid.float() * gt_weights_rep.reshape(-1, 1)
                elif gt_loss_type == 'gt':
                    gt_labels_rep = gt_labels_all
                    gt_weights_rep = pos_valid[:, 0, 0, :].float() * gt_weights.reshape(-1, 1)
                    gt_cls_prob = gt_cls_prob.reshape(num_gts, num_refine, -1)[:, 0]
                else:
                    raise ValueError()
                gt_labels = torch.full(gt_cls_prob.shape, 0., dtype=torch.float32).to(gt_cls_prob.device)
                gt_labels[torch.arange(len(gt_labels)), gt_labels_rep] = 1
                num_pos = max((gt_weights_rep > 0).sum(), 1)

                gt_loss = self.loss_mil.gfocal_loss(gt_cls_prob, gt_labels, gt_weights_rep)
                gt_loss = self.loss_cfg['gt_loss_weight'] * weight_reduce_loss(gt_loss, None, avg_factor=num_pos)
            losses['gt_loss'] = gt_loss

        if self.loss_cfg.get('with_mil_loss', True):
            refine_bag_policy = self.loss_cfg["refine_bag_policy"]
            if refine_bag_policy == 'independent_with_gt_bag':
                # treat num_refine as independent bags
                pos_pts = pos_pts.reshape(num_gts * num_refine, num_chosen, 3)
                pos_cls_outs = pos_cls_outs.reshape(num_gts * num_refine, num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts * num_refine, num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts * num_refine, num_chosen, -1)
                pos_weights = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                gt_labels_all = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
            elif refine_bag_policy == 'merge_to_gt_bag':
                pos_pts = pos_pts.reshape(num_gts, num_refine * num_chosen, 3)
                pos_cls_outs = pos_cls_outs.reshape(num_gts, num_refine * num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts, num_refine * num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts, num_refine * num_chosen, -1)
                pos_weights = gt_weights
            elif refine_bag_policy == 'only_refine_bag':
                si = 1 if num_refine > 1 else 0
                pos_pts = pos_pts[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, 3)
                pos_cls_outs = pos_cls_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_ins_outs = pos_ins_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_valid = pos_valid[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_weights = gt_weights
            else:
                raise ValueError
            pos_weights = pos_valid.float() * pos_weights.reshape(-1, 1, 1)

            self.random_remove(pos_pts, neg_pts, random_remove_rate=self.loss_cfg['random_remove_rate'])

            pos_cls_prob = self.get_cls_prob(pos_cls_outs)
            if dynamic_weight is not None:
                pos_loss, bag_acc, num_pos = self.loss_mil(pos_cls_prob, pos_ins_outs, gt_labels_all, pos_weights,
                                                           dynamic_weight.unsqueeze(-1))
            else:
                pos_loss, bag_acc, num_pos = self.loss_mil(pos_cls_prob, pos_ins_outs, gt_labels_all, pos_weights)
            losses.update({"pos_loss": pos_loss, "bag_acc": bag_acc})

        if self.loss_cfg.get("with_neg", True):
            neg_prob = self.get_cls_prob(neg_cls_outs)
            num_neg, num_class = neg_prob.shape

            neg_labels = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_prob.device)
            loss_weights = self.loss_cfg["neg_loss_weight"]
            neg_valid = neg_valid.reshape(num_neg, -1)

            neg_loss = self.loss_mil.gfocal_loss(neg_prob, neg_labels, neg_valid.float())
            if dynamic_weight is not None:
                neg_loss = loss_weights * dynamic_weight.mean() * weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
            else:
                neg_loss = loss_weights * weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
            losses.update({"neg_loss": neg_loss})
        #
        # if self.loss_cfg.get("with_pts_reg", True):
        #     pred_pts = pos_data.pred_pts[0]
        #     num_gen = pred_pts.shape[-2]
        #     pseudo_pts_label = torch.cat(pseudo_pts)[:, 0, :]
        #     pseudo_pts_label = pseudo_pts_label.unsqueeze(1).expand(num_gts, num_gen, 2)
        #     pred_pts = pred_pts.squeeze(1).reshape(-1, 2)
        #     pseudo_pts_label = pseudo_pts_label.reshape(-1, 2)
        #     # cls_valid =pos_cls_outs[torch.arange(num_gts),:,gt_labels_all,None].sigmoid()> self.pos_threshold
        #     merge_valid = torch.cat(merge_valid).unsqueeze(-1)
        #     pos_valid *= merge_valid
        #     pos_valid[:, 0, :] = True
        #     pos_valid = pos_valid.reshape(-1, 1)
        #     pos_cls_outs[torch.arange(num_gts), :, gt_labels_all]
        #     reg_loss = self.loss_reg(
        #         pred_pts / self.reg_norm,  # pts_out # (B, num_pts_all_lvl*k, 2)
        #         pseudo_pts_label / self.reg_norm,
        #         pos_valid,
        #         avg_factor=pos_valid.sum()
        #     )
        #     losses.update({"reg_loss": reg_loss})
        return losses

    def get_targets(self, anchor_pts, pred_pts, valid_flag_list, cls_feats_list, ins_feats_list, pseudo_boxes_list,
                    gt_points, gt_labels,
                    # label_weights,
                    img_metas, gt_points_ignore=None, unmap_outputs=True):
        """
        Args:
            pred_pts: (B, lvl*w*h*k, 2)
            valid_flag_list: (B, lvl*w*h*k)
            cls_outs_list: (B, lvl*w*h*k, num_cls_out)
            gt_points: [B, (num_gt, 2)]
            gt_labels: [B, (num_gt)]
            img_metas: [dict]
            gt_points_ignore:
            unmap_outputs:

        Returns:[B,(1280)],[B,(1280)],[B,(1280,2)]
        """
        num_imgs = len(img_metas)
        assert len(pred_pts) == num_imgs == len(valid_flag_list) == len(anchor_pts) == len(cls_feats_list) == len(
            ins_feats_list) == len(pseudo_boxes_list) == len(gt_points) == len(gt_labels) == len(img_metas)

        box_xywh = bbox_xyxy_to_cxcywh(torch.cat(pseudo_boxes_list))
        num_ins = ((torch.ceil(box_xywh[:, 2] / self.strides[0]) + 1) * (
                torch.ceil(box_xywh[:, 3] / self.strides[0]) + 1)).max().int()

        pos_data_list, neg_data_list = multi_apply(
            self._get_target_single, anchor_pts,
            pred_pts, valid_flag_list, cls_feats_list, ins_feats_list, pseudo_boxes_list,
            gt_points, gt_labels, img_metas, gt_points_ignore=gt_points_ignore,
            unmap_outputs=True, num_ins=np.int(num_ins))

        pos_data = PtAndFeat()
        neg_data = PtAndFeat()
        pos_data.img_len = len(pos_data_list)
        neg_data.img_len = len(neg_data_list)
        pos_data.cls_outs = [torch.cat([p['cls_outs'] for p in pos_data_list])]
        pos_data.ins_outs = [torch.cat([p['ins_outs'] for p in pos_data_list])]
        pos_data.pts = [torch.cat([p['anchor_pts_out'] for p in pos_data_list])]
        pos_data.valid = [torch.cat([p['pos_valid_mask'] for p in pos_data_list])]
        pos_data.pred_pts = [torch.cat([p['proposals_out'] for p in pos_data_list])]
        pos_data.cls_refine = [[p['cls_outs']] for p in pos_data_list]
        neg_data.cls_outs = [torch.cat([p['cls_outs'] for p in neg_data_list])]
        neg_data.pts = [torch.cat([p['neg_pts_out'] for p in neg_data_list])]
        neg_data.valid = [torch.cat([p['neg_valid_mask'] for p in neg_data_list])]
        neg_data.cls_refine = [[p['cls_outs']] for p in neg_data_list]

        pos_img_len = [[len(pts) for im_id, pts in enumerate(pos_pts_lvl)]
                       for lvl, pos_pts_lvl in enumerate([[p['anchor_pts_out'] for p in pos_data_list]])]
        neg_img_len = [[len(pts) for im_id, pts in enumerate(neg_pts_lvl)]
                       for lvl, neg_pts_lvl in enumerate([[p['neg_pts_out'] for p in neg_data_list]])]
        pos_data.img_len = pos_img_len
        neg_data.img_len = neg_img_len
        # neg_data['pts'] =
        # pos_data, neg_data = self.ecpl_2_cpr_form(cls_out_list, ins_out_list, neg_out_list, pos_data_list,
        #                                           neg_data_list)

        return pos_data, neg_data

    def bag_assign(self, anchor_pts, proposals, cls_feats, ins_feats, gt_points, gt_labels, img_metas,
                   gt_bboxes_ignore=None,
                   sample_r=7):
        num_gt = gt_labels.shape[0]
        dist = torch.cdist(gt_points, anchor_pts[..., :2], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        _, gt_index = dist.topk(k=1, largest=False)

        assigned_pts = (dist - sample_r * anchor_pts[None, ..., 2]) < 0
        assigned_pts_list = [torch.nonzero(p).squeeze(-1) for p in assigned_pts]

        # remove gt pts to fist
        gt_index = [i for i in gt_index]
        assigned_pts_list = [torch.cat([i, pts]) for i, pts in zip(gt_index, assigned_pts_list)]

        cls_feats_assigned = [cls_feats[ass] for ass in assigned_pts_list]
        ins_feats_assigned = [ins_feats[ass] for ass in assigned_pts_list]
        anchor_pts_assigned = [anchor_pts[ass] for ass in assigned_pts_list]
        proposals_assigned = [proposals[ass] for ass in assigned_pts_list]

        cls_feats_out = cls_feats.new_full((num_gt, 1, (sample_r * 2 + 1) ** 2, cls_feats.shape[-1]),
                                           0, dtype=torch.float)
        ins_feats_out = ins_feats.new_full((num_gt, 1, (sample_r * 2 + 1) ** 2, ins_feats.shape[-1]),
                                           0, dtype=torch.float)
        anchor_pts_out = anchor_pts.new_full((num_gt, 1, (sample_r * 2 + 1) ** 2, anchor_pts.shape[-1]),
                                             0, dtype=torch.float)
        proposals_out = proposals.new_full((num_gt, 1, (sample_r * 2 + 1) ** 2, proposals.shape[-1]),
                                           0, dtype=torch.float)
        pos_valid_mask = cls_feats.new_full((num_gt, 1, (sample_r * 2 + 1) ** 2, 1),
                                            0, dtype=torch.float)
        ## not cls wise

        for i, a in enumerate(cls_feats_assigned):
            cls_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(ins_feats_assigned):
            ins_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(anchor_pts_assigned):
            anchor_pts_out[i, 0, :len(a)] = a
        for i, a in enumerate(proposals_assigned):
            proposals_out[i, 0, :len(a)] = a
        for i, a in enumerate(cls_feats_assigned):
            pos_valid_mask[i, 0, :len(a)] = 1

        assigned_neg_pts = assigned_pts.sum(dim=0) == 0
        neg_valid_mask = assigned_neg_pts.unsqueeze(-1).expand((assigned_neg_pts.shape[0], self.num_classes))
        neg_cls_feats = cls_feats
        return dict(bag_cls_feats=cls_feats_out,
                    bag_ins_feats=ins_feats_out,
                    anchor_pts_out=anchor_pts_out,
                    proposals_out=proposals_out,
                    pos_valid_mask=pos_valid_mask), \
               dict(neg_cls_feats=neg_cls_feats,
                    neg_pts_out=anchor_pts,
                    neg_valid_mask=neg_valid_mask)

    def bag_assign_1(self, anchor_pts, proposals, cls_feats, ins_feats, pseudo_boxes, gt_points, gt_labels, img_metas,
                     gt_bboxes_ignore=None,
                     sample_r=7, num_ins=None):
        num_gt = gt_labels.shape[0]
        proposals = proposals[:, :2]
        anchor_pts_cor = anchor_pts[..., :2]
        v1 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 0]
        v2 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 2]
        v3 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 1]
        v4 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 3]
        ass = (v1 * v2 * v3 * v4).permute(1, 0)

        dist = torch.cdist(gt_points, anchor_pts[..., :2], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        _, gt_index = dist.topk(k=1, largest=False)

        assigned_pts = (dist - sample_r * anchor_pts[None, ..., 2]) < 0
        assigned_pts_list = [torch.nonzero(p).squeeze(-1) for p in ass]

        # remove gt pts to fist
        gt_index = [i for i in gt_index]
        assigned_pts_list = [torch.cat([i, pts]) for i, pts in zip(gt_index, assigned_pts_list)]

        cls_feats_assigned = [cls_feats[ass] for ass in assigned_pts_list]
        ins_feats_assigned = [ins_feats[ass] for ass in assigned_pts_list]
        anchor_pts_assigned = [anchor_pts[ass] for ass in assigned_pts_list]
        proposals_assigned = [proposals[ass] for ass in assigned_pts_list]

        cls_feats_out = cls_feats.new_full((num_gt, 1, num_ins, cls_feats.shape[-1]),
                                           0, dtype=torch.float)
        ins_feats_out = ins_feats.new_full((num_gt, 1, num_ins, ins_feats.shape[-1]),
                                           0, dtype=torch.float)
        anchor_pts_out = anchor_pts.new_full((num_gt, 1, num_ins, anchor_pts.shape[-1]),
                                             0, dtype=torch.float)
        proposals_out = proposals.new_full((num_gt, 1, num_ins, proposals.shape[-1]),
                                           0, dtype=torch.float)
        pos_valid_mask = cls_feats.new_full((num_gt, 1, num_ins, 1),
                                            0, dtype=torch.float)
        ## not cls wise

        for i, a in enumerate(cls_feats_assigned):
            cls_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(ins_feats_assigned):
            ins_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(anchor_pts_assigned):
            anchor_pts_out[i, 0, :len(a)] = a
        for i, a in enumerate(proposals_assigned):
            proposals_out[i, 0, :len(a)] = a
        for i, a in enumerate(cls_feats_assigned):
            pos_valid_mask[i, 0, :len(a)] = 1

        assigned_neg_pts = ass.sum(dim=0) == 0
        neg_valid_mask = assigned_neg_pts.unsqueeze(-1).expand((assigned_neg_pts.shape[0], self.num_classes))
        neg_cls_feats = cls_feats
        return dict(bag_cls_feats=cls_feats_out,
                    bag_ins_feats=ins_feats_out,
                    anchor_pts_out=anchor_pts_out,
                    proposals_out=proposals_out,
                    pos_valid_mask=pos_valid_mask), \
               dict(neg_cls_feats=neg_cls_feats,
                    neg_pts_out=anchor_pts,
                    neg_valid_mask=neg_valid_mask)

    def bag_assign_2(self, anchor_pts, proposals, cls_feats, ins_feats, pseudo_boxes, gt_points, gt_labels, img_metas,
                     gt_bboxes_ignore=None,
                     sample_r=7, num_ins=None):
        pseudo_boxes = bbox_xyxy_to_cxcywh(pseudo_boxes)
        pseudo_boxes[:, 2] *= 1.2
        pseudo_boxes[:, 3] *= 1.2
        pseudo_boxes = bbox_cxcywh_to_xyxy(pseudo_boxes)
        num_gt = gt_labels.shape[0]
        proposals = proposals[:, :2]
        anchor_pts_cor = anchor_pts[..., :2]
        v1 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 0]
        v2 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 2]
        v3 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 1]
        v4 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 3]
        ass = (v1 * v2 * v3 * v4).permute(1, 0)

        dist = torch.cdist(gt_points, anchor_pts[..., :2], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        _, gt_index = dist.topk(k=1, largest=False)

        assigned_pts = (dist - sample_r * anchor_pts[None, ..., 2]) < 0
        assigned_pts_list_ori = [torch.nonzero(p).squeeze(-1) for p in ass]

        # remove gt pts to fist
        gt_index = [i for i in gt_index]
        assigned_pts_list_ori = [torch.cat([i, pts]) for i, pts in zip(gt_index, assigned_pts_list_ori)]

        assigned_pts_list = [np.random.choice(range(len(i)), 100) for i in assigned_pts_list_ori]  ##random
        num_ins = 100
        assigned_pts_list = torch.tensor(assigned_pts_list).to(gt_labels.device)
        assigned_pts_list = [assigned_pts_list_ori[i][j] for i, j in enumerate(assigned_pts_list)]

        cls_feats_assigned = [cls_feats[ass] for ass in assigned_pts_list]
        ins_feats_assigned = [ins_feats[ass] for ass in assigned_pts_list]
        anchor_pts_assigned = [anchor_pts[ass] for ass in assigned_pts_list]
        proposals_assigned = [proposals[ass] for ass in assigned_pts_list]

        cls_feats_out = cls_feats.new_full((num_gt, 1, num_ins, cls_feats.shape[-1]),
                                           0, dtype=torch.float)
        ins_feats_out = ins_feats.new_full((num_gt, 1, num_ins, ins_feats.shape[-1]),
                                           0, dtype=torch.float)
        anchor_pts_out = anchor_pts.new_full((num_gt, 1, num_ins, anchor_pts.shape[-1]),
                                             0, dtype=torch.float)
        proposals_out = proposals.new_full((num_gt, 1, num_ins, proposals.shape[-1]),
                                           0, dtype=torch.float)
        pos_valid_mask = cls_feats.new_full((num_gt, 1, num_ins, 1),
                                            0, dtype=torch.float)
        ## not cls wise

        for i, a in enumerate(cls_feats_assigned):
            cls_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(ins_feats_assigned):
            ins_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(anchor_pts_assigned):
            anchor_pts_out[i, 0, :len(a)] = a
        for i, a in enumerate(proposals_assigned):
            proposals_out[i, 0, :len(a)] = a
        for i, a in enumerate(cls_feats_assigned):
            pos_valid_mask[i, 0, :len(a)] = 1

        # assigned_neg_pts = ass.sum(dim=0) == 0
        neg_valid_mask = cls_feats.new_full((cls_feats.shape[0], self.num_classes),
                                            0, dtype=torch.float)
        for i in range(len(ass)):
            neg_valid_mask[:, gt_labels[i]] += ass[i]
        neg_valid_mask = neg_valid_mask == 0
        # neg_valid_mask = assigned_neg_pts.unsqueeze(-1).expand((assigned_neg_pts.shape[0], self.num_classes))
        neg_cls_feats = cls_feats
        return dict(bag_cls_feats=cls_feats_out,
                    bag_ins_feats=ins_feats_out,
                    anchor_pts_out=anchor_pts_out,
                    proposals_out=proposals_out,
                    pos_valid_mask=pos_valid_mask), \
               dict(neg_cls_feats=neg_cls_feats,
                    neg_pts_out=anchor_pts,
                    neg_valid_mask=neg_valid_mask)

    def bag_assign_3(self, anchor_pts, proposals, cls_feats, ins_feats, pseudo_boxes, gt_points, gt_labels, img_metas,
                     gt_bboxes_ignore=None,
                     sample_r=7, num_ins=None):
        pseudo_boxes = bbox_xyxy_to_cxcywh(pseudo_boxes)
        pseudo_boxes[:, 2] *= 1.2
        pseudo_boxes[:, 3] *= 1.2
        pseudo_boxes = bbox_cxcywh_to_xyxy(pseudo_boxes)
        num_gt = gt_labels.shape[0]
        proposals = proposals[:, :2]
        anchor_pts_cor = anchor_pts[..., :2]
        v1 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 0]
        v2 = anchor_pts_cor[:, 0, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 2]
        v3 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) >= pseudo_boxes[:, 1]
        v4 = anchor_pts_cor[:, 1, None].expand(anchor_pts_cor.shape[0], num_gt) <= pseudo_boxes[:, 3]
        ass = (v1 * v2 * v3 * v4).permute(1, 0)

        dist = torch.cdist(gt_points, anchor_pts[..., :2], p=2, compute_mode='donot_use_mm_for_euclid_dist')
        _, gt_index = dist.topk(k=1, largest=False)

        assigned_pts = (dist - sample_r * anchor_pts[None, ..., 2]) < 0
        assigned_pts_list_ori = [torch.nonzero(p).squeeze(-1) for p in ass]

        # remove gt pts to fist
        gt_index = [i for i in gt_index]
        assigned_pts_list_ori = [torch.cat([i, pts]) for i, pts in zip(gt_index, assigned_pts_list_ori)]

        assigned_pts_list = [np.random.choice(range(len(i)), 100) for i in assigned_pts_list_ori]  ##random
        num_ins = 100
        assigned_pts_list = torch.tensor(assigned_pts_list).to(gt_labels.device)
        assigned_pts_list = [assigned_pts_list_ori[i][j] for i, j in enumerate(assigned_pts_list)]

        cls_feats_assigned = [cls_feats[ass] for ass in assigned_pts_list]
        ins_feats_assigned = [ins_feats[ass] for ass in assigned_pts_list]
        anchor_pts_assigned = [anchor_pts[ass] for ass in assigned_pts_list]
        proposals_assigned = [proposals[ass] for ass in assigned_pts_list]

        cls_feats_out = cls_feats.new_full((num_gt, 1, num_ins, cls_feats.shape[-1]),
                                           0, dtype=torch.float)
        ins_feats_out = ins_feats.new_full((num_gt, 1, num_ins, ins_feats.shape[-1]),
                                           0, dtype=torch.float)
        anchor_pts_out = anchor_pts.new_full((num_gt, 1, num_ins, anchor_pts.shape[-1]),
                                             0, dtype=torch.float)
        proposals_out = proposals.new_full((num_gt, 1, num_ins, proposals.shape[-1]),
                                           0, dtype=torch.float)
        pos_valid_mask = cls_feats.new_full((num_gt, 1, num_ins, 1),
                                            0, dtype=torch.float)
        ## not cls wise

        for i, a in enumerate(cls_feats_assigned):
            cls_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(ins_feats_assigned):
            ins_feats_out[i, 0, :len(a)] = a
        for i, a in enumerate(anchor_pts_assigned):
            anchor_pts_out[i, 0, :len(a)] = a
        for i, a in enumerate(proposals_assigned):
            proposals_out[i, 0, :len(a)] = a
        for i, a in enumerate(cls_feats_assigned):
            pos_valid_mask[i, 0, :len(a)] = 1

        # assigned_neg_pts = ass.sum(dim=0) == 0
        neg_valid_mask = cls_feats.new_full((cls_feats.shape[0], self.num_classes),
                                            0, dtype=torch.float)
        for i in range(len(ass)):
            neg_valid_mask[:, gt_labels[i]] += ass[i]
        neg_valid_mask = neg_valid_mask == 0
        # neg_valid_mask = assigned_neg_pts.unsqueeze(-1).expand((assigned_neg_pts.shape[0], self.num_classes))
        neg_cls_feats = cls_feats
        return dict(bag_cls_feats=cls_feats_out,
                    bag_ins_feats=ins_feats_out,
                    anchor_pts_out=anchor_pts_out,
                    proposals_out=proposals_out,
                    pos_valid_mask=pos_valid_mask), \
               dict(neg_cls_feats=neg_cls_feats,
                    neg_pts_out=anchor_pts,
                    neg_valid_mask=neg_valid_mask)

    def _get_target_single(self, anchor_pts, pred_pts, valid_flags, cls_feats, ins_feats, pseudo_boxes, gt_points,
                           gt_labels,
                           img_metas,
                           gt_points_ignore=None,
                           unmap_outputs=True,
                           num_ins=None):
        """
        Args:
            pred_pts: (lvl*w*h*k, 2)
            valid_flags: (lvl*w*h*k)
            cls_outs: (lvl*w*h*k, num_cls_out) TODO: k >>> init_layer
            gt_points: (num_gt, 2)
            gt_labels: (num_gt)
            img_metas: dict
            gt_points_ignore:
            unmap_outputs:

        Returns:
        """
        # inside_flags = valid_flags  # lvl*w*h*k [0]
        #
        # if not inside_flags.any():
        #     return (None,) * 3

        # proposals = pred_pts[inside_flags, :]
        # cls_pred = cls_feats[inside_flags, :]
        # ins_pred = ins_feats[inside_flags, :]
        pos_data, neg_data = self.bag_assign_3(anchor_pts, pred_pts, cls_feats, ins_feats, pseudo_boxes, gt_points,
                                               gt_labels, img_metas,
                                               gt_bboxes_ignore=None, sample_r=self.sample_r, num_ins=num_ins)
        bag_cls_feats = pos_data['bag_cls_feats']
        bag_ins_feats = pos_data['bag_cls_feats']
        neg_cls_feats = neg_data['neg_cls_feats']
        cls_o, ins_o = self.get_pts_outs(bag_cls_feats, bag_ins_feats)
        neg_o = self.get_pts_outs(neg_cls_feats)
        #
        # labels, label_weights, bbox_gt, proposals_weights = self.sample_result_to_target(proposals, sampling_result,
        #                                                                                  gt_labels)
        # # map up to original set of proposals 映射到
        # if unmap_outputs:
        #     do_unmap = partial(unmap, count=pred_pts.size(0), inds=inside_flags)  # lvl*w*h*k
        #     labels, label_weights, bbox_gt, proposals_weights = map(do_unmap, [labels, label_weights,
        #                                                                        bbox_gt, proposals_weights])
        pos_data['cls_outs'] = cls_o
        pos_data['ins_outs'] = ins_o
        neg_data['cls_outs'] = neg_o

        return pos_data, neg_data

    def sample_result_to_target(self, proposals, sampling_result, gt_labels):
        num_valid_proposals, loc_s = proposals.shape
        bbox_gt = proposals.new_zeros([num_valid_proposals, loc_s])
        proposals_weights = proposals.new_zeros([num_valid_proposals, loc_s])
        labels = proposals.new_full((num_valid_proposals,), self.num_classes, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds  # 15
        neg_inds = sampling_result.neg_inds  # 305

        neg_weight = self.train_cfg.get('neg_weight', 1.0)
        pos_weight = self.train_cfg.get('pos_weight', 1.0)
        if len(pos_inds) > 0:
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes
            proposals_weights[pos_inds, :] = 1.0
            # Only rpn gives gt_labels as None Foreground is the first class
            labels[pos_inds] = 0 if gt_labels is None else gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0 if neg_weight <= 0 else neg_weight
        return labels, label_weights, bbox_gt, proposals_weights

    def get_bboxes(self, cls_feats, ins_feats, pts_outs, img_metas, cfg=None, rescale=False, with_nms=True):
        assert len(cls_feats) == len(pts_outs)
        anchor_pts, pred_pts, valid_flag, cls_feats, ins_feats = self.get_pred_points(cls_feats, ins_feats, pts_outs,
                                                                                      img_metas)
        cls_outs = self.get_pts_outs(cls_feats)
        # cls_outs = self.get_cls_prob(cls_outs)
        # (B, num_pts_all_lvl * k, 2),(B, lvl * w * h*k),(B, lvl * w * h * k, num_cls_out)
        result_list = []
        for img_id in range(len(img_metas)):
            # ##
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            point_scores, labels = self._get_bboxes_single(pred_pts[img_id][..., :2], valid_flag[img_id],
                                                           cls_outs[img_id],
                                                           img_shape, scale_factor, cfg, rescale)
            bbox_scores = self.center_to_pseudo_bbox([point_scores])[0]
            result_list.append((bbox_scores, labels))
        return result_list

    def _get_bboxes_single(self, pred_pts, valid_flag, cls_outs, img_shape, scale_factor, cfg, rescale=False,
                           with_nms=True):
        """
        Args:
            pred_pts: (lvl*w*h*k, 2)
            valid_flag: (lvl*w*h*k),
            cls_outs: (lvl*w*h*k, num_cls_out)
            cfg:
            rescale:
        Returns:
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert pred_pts.shape[-1] == 2, pred_pts.shape
        pred_pts = pred_pts.reshape(len(self.strides), -1, pred_pts.shape[-1])
        cls_outs = cls_outs.reshape(len(self.strides), -1, self.num_cls_out)
        assert len(cls_outs) == len(pred_pts)
        mlvl_points = []
        mlvl_scores = []
        for i_lvl, (cls_score, points_pred) in enumerate(zip(cls_outs, pred_pts)):  # (w*h*k,num_cls_out),(w*h*k,2)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)
            nms_pre = cfg.get('nms_pre', -1)  # 1000
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]. since mmdet v2.0. BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)  # ### 层内topk
                scores = scores[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
            x = points_pred[:, 0].clamp(min=0, max=img_shape[1])  # 拉回来
            y = points_pred[:, 1].clamp(min=0, max=img_shape[0])
            points = torch.stack([x, y], dim=-1)
            mlvl_points.append(points)
            mlvl_scores.append(scores)
        mlvl_points = torch.cat(mlvl_points)  # (num_lvl*nms_pre,2)
        mlvl_scores = torch.cat(mlvl_scores)  # (num_lvl*nms_pre,80)

        if rescale:
            mlvl_points /= mlvl_points.new_tensor(scale_factor[:2])
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # TODO: cfg.score_thr
        if with_nms:
            dets = torch.cat([mlvl_points, mlvl_points.new_zeros([mlvl_scores.shape[0], 1])], dim=1)
            pseudo_boxes = self.center_to_pseudo_bbox([dets])[0]

            det_bboxes, det_labels = multiclass_nms(pseudo_boxes[:, :-1], mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            det_score = det_bboxes[:, 4, None]
            from mmdet.core.bbox import bbox_xyxy_to_cxcywh
            det_bboxes = bbox_xyxy_to_cxcywh(det_bboxes[:, :-1])
            det_bboxes = torch.cat([det_bboxes[:, :2], det_score], dim=1)
            return det_bboxes, det_labels
        else:
            mlvl_points = mlvl_points[:, None].expand(  # (n,2)>>(n,80,2)
                mlvl_points.size(0), self.num_cls_out, mlvl_points.shape[-1])
            labels = torch.arange(self.num_cls_out, dtype=torch.long)
            labels = labels.view(1, -1).expand_as(mlvl_scores)
            mlvl_points = mlvl_points.reshape(-1, mlvl_points.shape[-1])
            mlvl_scores = mlvl_scores.reshape(-1)
            labels = labels.reshape(-1)

            valid_mask = mlvl_scores > cfg.score_thr  # ### 层间进行得分的阈值
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            points, scores, labels = mlvl_points[inds], mlvl_scores[inds], labels[inds]
            dets = torch.cat([points, scores[:, None]], -1)

            if 0 < cfg.max_per_img < len(scores):
                _, idx = scores.topk(cfg.max_per_img, largest=True)
                dets, labels = dets[idx], labels[idx]
            return dets, labels

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes. [reppoints]

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes. [lvl, (h, w)]
            img_metas (list[dict]): Image meta info. [b]

        Returns:
            tuple: points of each image, valid flags of each image
                    [B, lvl, (w*h, 3)]
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.strides[i], device)  # (w*h,3)
            # points = points[:, :2]  # [shift_xx, shift_yy, stride]
            multi_level_points.append(points)  # [lvl, (w*h,3)]
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]  # [B, lvl, (w*h,3)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def pseudo_bbox_to_center(self, gt_bboxes):
        """
        Transform pseudo bbox to center point
        Args:
            gt_bboxes: [num_imgs, (num_pts, 2)]
        Returns:
        """
        return [(gt_bboxes_img[:, :2] + gt_bboxes_img[:, 2:]) / 2
                for gt_bboxes_img in gt_bboxes]

    def center_to_pseudo_bbox(self, center_scores):
        """
        center_scores: [B, (N, 3)], 3 is (cx, cy, score)
        Returns:
        """
        pseudo_wh = self.test_cfg.get('pseudo_wh', (16, 16))
        pseudo_wh = center_scores[0].new_tensor(pseudo_wh)
        return [torch.cat([center[:, :2] - pseudo_wh / 2, center[:, :2] + pseudo_wh / 2, center[:, 2:]], dim=-1)
                for center in center_scores]

    def get_clsandpts(self, cls_feats, ins_feats, pts_outs, img_metas):
        anchor_pts, pred_pts, valid_flag, cls_feats, ins_feats = self.get_pred_points(cls_feats, ins_feats, pts_outs,
                                                                                      img_metas)

        pass

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        if hasattr(self, '_get_bboxes'):
            gbs_sig = signature(self._get_bboxes)
        else:
            gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            # outs = self.get_clsandpts(*outs,img_meta)

            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]

            ### after nms
            # _bboxes (*5)
            _bboxes = bbox_outputs[0]
            _labels = bbox_outputs[1]
            _scores = _bboxes.new_full((_bboxes.shape[0], self.num_classes), 0)
            _scores[torch.arange(_bboxes.shape[0]), _labels] = _bboxes[:, 4]
            aug_bboxes.append(_bboxes[:, :4])
            aug_scores.append(_scores)

            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
            merged_scores = torch.cat([merged_scores, padding], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]


class TestP2PHead(object):
    DO_TEST = False
    count = 0

    @staticmethod
    def test_assign(img_metas, proposal_points, matched_gt_points, proposal_weight):
        if not TestP2PHead.DO_TEST:
            return
        TestP2PHead.count += 1
        if TestP2PHead.count > 10:
            exit(-1)

        def to_numpy(data):
            data = data[0]
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data[0], torch.Tensor):
                return [d.detach().cpu().numpy() for d in data]

        img_path = img_metas[0]['filename']
        all_proposal_points = to_numpy(proposal_points)
        valid = (proposal_weight[0][:, 0] > 0) & (proposal_weight[0][:, 1] > 0)
        proposal_points = proposal_points[0][valid]
        matched_gt_points = matched_gt_points[0][valid]
        proposal_points = to_numpy([proposal_points])
        matched_gt_points = to_numpy([matched_gt_points])

        from PIL import Image
        import matplotlib.pyplot as plt
        from huicv.vis.visualize import get_hsv_colors, draw_a_bbox
        from huicv.plot_paper.plt_paper_config import set_plt
        import os

        colors = get_hsv_colors(80)
        img = np.array(Image.open(img_path))
        plt.figure(figsize=(14, 8))

        fig = set_plt(plt)
        plt.imshow(img)
        plt.scatter(all_proposal_points[:, 0], all_proposal_points[:, 1], s=40, c='b')
        plt.scatter(proposal_points[:, 0], proposal_points[:, 1], s=40, c='g')
        plt.scatter(matched_gt_points[:, 0], matched_gt_points[:, 1], s=40, c='r')
        for i in range(len(proposal_points)):
            plt.plot([proposal_points[i][0], matched_gt_points[i][0]], [proposal_points[i][1], matched_gt_points[i][1]],
                     '--', color=(0, 0, 0))

        img_name = os.path.split(img_path)[-1]
        plt.savefig("exp/debug/P2P/vis_{}".format(img_name))
        plt.show()


class Statistic(object):
    def sum(self, name, x):
        if not hasattr(self, name):
            setattr(self, f"{name}", [0, 0])
        value, count = getattr(self, f"{name}")
        value = value + x.sum(dim=-1)
        count += len(x)
        setattr(self, name, [value, count])
        return value, count

    def mean(self, name, x):
        s, c = self.sum(name, x)
        return s / c

    def print_mean(self, name, x):
        print(name, self.mean(name, x))


class TestCPRHead(object):
    DO_TEST = True

    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    _INSTANCE = {}

    @staticmethod
    def get(obj):
        assert isinstance(obj, PointRefiner)
        if obj not in TestCPRHead._INSTANCE:
            TestCPRHead._INSTANCE[obj] = test_cpr = TestCPRHead(len(TestCPRHead._INSTANCE))
            return test_cpr
        return TestCPRHead._INSTANCE[obj]

    def __init__(self, id):
        self.id = id
        self.sta = Statistic()
        self.count = 0

    def test_extract_point_feat(self, pts_in_feat, feat, pts_feats):
        if not TestCPRHead.DO_TEST:
            return
        pts_in_feat = pts_in_feat.reshape(-1, 2)
        x, y = pts_in_feat[:, 0], pts_in_feat[:, 1]
        idx_pts_feat = feat[0][:, y.long(), x.long()].permute(1, 0)
        pts_feats = pts_feats.reshape(-1, pts_feats.shape[-1])
        s = (idx_pts_feat - pts_feats).abs().max()
        if s > 1e-4:
            print("[test_extract_point_feat]:", s, x, y, feat.shape)

    def test_chosen_fpn_level(self, prob_bag_cls, prob_bag_ins, lvl_max_prob_idx,
                              all_gt_idx, all_gt_labels, num_all_gts, num_level):
        if not TestCPRHead.DO_TEST:
            return
        if num_level == 1:
            return
        sta = self.sta

        prob_bag_cls_l = prob_bag_cls[all_gt_idx, :, all_gt_labels]  # (num_gt, num_samples_all_lvl)
        prob_bag_ins_l = prob_bag_ins[all_gt_idx, :, all_gt_labels]
        for k, d in enumerate([prob_bag_cls_l, prob_bag_ins_l, (prob_bag_cls_l * prob_bag_ins_l)]):
            d = d.reshape(num_all_gts, num_level, -1)
            other_lvl_d = []
            chose_lvl_d = []
            for i in range(num_all_gts):
                for j in range(num_level):
                    if j == lvl_max_prob_idx[i]:
                        chose_lvl_d.append(d[i, j])
                    else:
                        other_lvl_d.append(d[i, j])
            chose_lvl_d, other_lvl_d = torch.stack(chose_lvl_d), torch.stack(other_lvl_d)
            print(f'{k} lvl data mean %.1e, %.1e, %.1e, %.1e, [%.1e, %.1e, %.1e, %.1e]' % ((
                                                                                               other_lvl_d.mean().item(),
                                                                                               chose_lvl_d.mean().item(),
                                                                                               chose_lvl_d.mean().item() / other_lvl_d.mean().item(),
                                                                                               (
                                                                                                       chose_lvl_d.max() / chose_lvl_d.mean()).item()) +
                                                                                           tuple(d.mean(dim=(0,
                                                                                                             2)).detach().cpu().numpy().tolist())))

        assert len(prob_bag_cls.shape) == 3
        prob_bag_cls_max, max_cls_idx = prob_bag_cls_l.max(dim=-1)
        prob_bag_ins_max, max_ins_idx = prob_bag_ins_l.max(dim=-1)
        print("max_i(cls)==max_i(ins)", sta.mean('max_i(cls)==max_i(ins)', (max_cls_idx == max_ins_idx).float()).item())
        rank = (prob_bag_ins_l >= prob_bag_ins_l[all_gt_idx, max_cls_idx].unsqueeze(dim=1)).float().sum(dim=1)
        print("ins rank of cls max", sta.mean('ins rank of cls max', rank).item())
        rank = (prob_bag_cls_l >= prob_bag_cls_l[all_gt_idx, max_ins_idx].unsqueeze(dim=1)).float().sum(dim=1)
        print("cls rank of ins max", sta.mean('cls rank of ins max', rank).item())
        x = prob_bag_cls_l[all_gt_idx, max_ins_idx] / prob_bag_cls_max
        print("cls[ins_max_idx] / cls_max", sta.mean('cls[ins_max_idx] / cls_max', x).item())
        x = prob_bag_ins_l[all_gt_idx, max_cls_idx] / prob_bag_ins_max
        print("ins[cls_max_idx] / ins_max", sta.mean('ins[cls_max_idx] / ins_max', x).item())

    def test_refine_point2(self, points, gt_true_bboxes, not_refine):
        points, gt_true_bboxes, not_refine = torch.cat(points), torch.cat(gt_true_bboxes), torch.cat(not_refine)
        inside = (gt_true_bboxes[:, 0] < points[:, 0]) & (points[:, 0] < gt_true_bboxes[:, 2]) & \
                 (gt_true_bboxes[:, 1] < points[:, 1]) & (points[:, 1] < gt_true_bboxes[:, 3])
        outside = inside.logical_not()
        outside_bboxes = gt_true_bboxes[outside]
        outside_size = ((outside_bboxes[:, 2] - outside_bboxes[:, 0]) * (
                outside_bboxes[:, 3] - outside_bboxes[:, 1])) ** 0.5
        sta = self.sta
        print("id", self.id)
        print("refine rate", sta.mean("refine rate", not_refine.logical_not().float()).item())
        print("outside rate", sta.mean("outside rate", outside.float()).item(),
              "outside size", sta.mean("outside size", outside_size).item())
        print()

    def test_grid(self, grid_data_list, gt_labels, img_metas):
        def pad_img(img, pad_shape):
            pad_img = np.zeros(pad_shape).astype(img.dtype)
            pad_img[:img.shape[0], :img.shape[1]] = img
            return pad_img

        def mask_img(mask, img):
            # return mask
            cmap = plt.get_cmap('jet')
            h, w = img.shape[:2]
            heatmap = Image.fromarray((cmap(mask)[..., :3] * 255).astype(np.uint8))
            heatmap = np.array(heatmap.resize((w, h))) / 255
            return (heatmap + img) / 2

        def plt_heatmap(grid_cls_prob, pos_labels, neg_labels, img, pad_shape):
            # grid_cls_prob[0, 0] = grid_cls_prob.max()
            # grid_cls_prob[-1, -1] = grid_cls_prob.min()
            img = np.array(img).astype(np.float32) / 255
            img = pad_img(img, pad_shape)

            k = 3
            plt.figure(figsize=(12, 6))
            for i, l in enumerate(pos_labels[:k]):
                plt.subplot(2, k, i + 1)
                plt.imshow(mask_img(grid_cls_prob[:, :, l], img))
                max_score = grid_cls_prob[:, :, l].max().round(2)
                plt.title(f"pos: {l}({TestCPRHead.CLASSES[l]}); max_score: {str(max_score)}")
            for i, l in enumerate(neg_labels[:k]):
                plt.subplot(2, k, k + i + 1)
                plt.imshow(mask_img(grid_cls_prob[:, :, l], img))
                max_score = grid_cls_prob[:, :, l].max().round(2)
                plt.title(f"neg: {l}({TestCPRHead.CLASSES[l]}); max_score: {str(max_score)}")
            plt.show()

        import matplotlib.pyplot as plt
        from PIL import Image
        grid_cls_prob = grid_data_list[0].cls_prob[0].cpu().numpy()
        grid_valid = grid_data_list[0].valid[0].float().cpu().numpy()
        labels = gt_labels[0].cpu().numpy()
        img_meta = img_metas[0]

        pos_labels = set(labels.tolist())
        neg_labels = list(set(list(range(80))) - pos_labels)
        from random import shuffle
        shuffle(neg_labels)
        pos_labels = list(pos_labels)

        img = Image.open(img_meta['filename'])
        plt_heatmap(grid_cls_prob, pos_labels, neg_labels, img, img_meta['pad_shape'])
        plt_heatmap(grid_valid, pos_labels, neg_labels, img, img_meta['pad_shape'])

    def test_refine_point(self, points, chosen_lvl, chosen_pts_all, gt_r_points, gt_labels, img_metas, gt_true_bboxes,
                          not_refine, fmt_points_score):

        if not TestCPRHead.DO_TEST:
            return
        self.count += 1
        # if self.count < 1001:
        #     return
        if self.count > 1000:  ##0-30
            exit(-1)

        def to_numpy(data):
            data = data[0]
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data[0], torch.Tensor):
                return [d.detach().cpu().numpy() for d in data]

        def inside(points, gt_true_bboxes):
            is_inside = (gt_true_bboxes[:, 0] < points[:, 0]) & (points[:, 0] < gt_true_bboxes[:, 2]) & \
                        (gt_true_bboxes[:, 1] < points[:, 1]) & (points[:, 1] < gt_true_bboxes[:, 3])
            return is_inside

        self.test_refine_point2(points, gt_true_bboxes, not_refine)

        img_path = img_metas[0]['filename']
        sw, sh = img_metas[0]['scale_factor'][:2]
        img_gt_r_points = to_numpy(gt_r_points)
        img_points = to_numpy(points)
        img_gt_labels = to_numpy(gt_labels)
        img_true_bboxes = to_numpy(gt_true_bboxes) if gt_true_bboxes is not None else None
        img_chosen_pts = to_numpy(chosen_pts_all)
        img_chosen_lvl = to_numpy(chosen_lvl) if chosen_lvl is not None else None
        img_not_refine = to_numpy(not_refine) if not_refine is not None else None
        img_scores = to_numpy(fmt_points_score) if fmt_points_score is not None else None

        img_gt_points = img_gt_r_points[:, 0]
        if img_true_bboxes is not None:
            assert len(img_true_bboxes) == len(img_gt_points), f"{len(img_true_bboxes)} vs {len(img_gt_points)}"

        from PIL import Image
        import matplotlib.pyplot as plt
        from huicv.vis.visualize import get_hsv_colors, draw_a_bbox
        from huicv.plot_paper.plt_paper_config import set_plt
        import os

        colors = get_hsv_colors(80)
        colors = [(0.65, 0.65, 0.65)] * 80  # 全黑色
        img = Image.open(img_path)
        w, h = img.width, img.height
        img = img.resize((round(int(w * sw)), int(round(h * sh))))
        img = np.array(img)
        plt.figure(figsize=(14, 8))

        fig = set_plt(plt)
        plt.imshow(img)
        plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=5, c='#22fe61',
                    zorder=11)  ## coarse point color 22fe61 ‘g’
        for i in range(len(img_points)):
            # draw true bbox
            if img_true_bboxes is not None:
                draw_a_bbox(img_true_bboxes[i], color=colors[img_gt_labels[i]])
            # link gt_pt -- refine_pt
            plt.plot([img_points[i, 0], img_gt_points[i, 0]], [img_points[i, 1], img_gt_points[i, 1]], '--',
                     linewidth=3, color=colors[img_gt_labels[i]])
            if img_not_refine is None or not img_not_refine[i]:
                # refine_pt -- chosen_pts
                for j in range(len(img_chosen_pts[i])):
                    p1, p2 = img_gt_points[i], img_chosen_pts[i][j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], ':',
                             color=colors[img_gt_labels[i]])  ##换虚线连接'--', color=colors[img_gt_labels[i]]
                # chosen_pts
                plt.scatter(img_chosen_pts[i][:, 0], img_chosen_pts[i][:, 1], s=5, c='#ed7d31',
                            zorder=10)  ## semantic point color:orange ed7d31 'b'
            # chosen fpn level
            if img_chosen_lvl is not None:
                plt.text(img_points[i][0], img_points[i][1], s=f"{img_chosen_lvl[i]}", fontsize=10)
            if img_scores is not None:
                plt.text(img_points[i][0], img_points[i][1], s=f"{(img_scores[i] * 100).round(2)}", color=(1, 1, 1),
                         fontsize=10)
        # plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=40, c='g')
        is_inside = inside(img_points, img_true_bboxes)
        plt.scatter(img_points[is_inside, 0], img_points[is_inside, 1], s=5, c='#ffff00',
                    zorder=10)  ## refined point color:yellow, size 'r'
        plt.scatter(img_points[np.logical_not(is_inside), 0], img_points[np.logical_not(is_inside), 1], s=40,
                    c=(0, 0, 0))
        img_name = os.path.split(img_path)[-1]
        plt.savefig("exp/debug/CPR/vis_{}".format(img_name))
        plt.show()
        plt.clf()

        fig = set_plt(plt)
        plt.imshow(img)
        plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=5, c='#22fe61', zorder=11)  # g
        for i in range(len(img_points)):
            # draw true bbox
            if img_true_bboxes is not None:
                draw_a_bbox(img_true_bboxes[i], color=colors[img_gt_labels[i]])
            # link gt_pt -- refine_pt
            # plt.plot([img_points[i, 0], img_gt_points[i, 0]], [img_points[i, 1], img_gt_points[i, 1]], '--',
            #          linewidth=3, color=colors[img_gt_labels[i]])
            # if img_not_refine is None or not img_not_refine[i]:
            #     # chosen_pts
            #     plt.scatter(img_chosen_pts[i][:, 0], img_chosen_pts[i][:, 1], s=20, c='b')
            #     # refine_pt -- chosen_pts
            #     for j in range(len(img_chosen_pts[i])):
            #         p1, p2 = img_gt_points[i], img_chosen_pts[i][j]
            #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[img_gt_labels[i]])
            # # chosen fpn level
            # if img_chosen_lvl is not None:
            #     plt.text(img_points[i][0], img_points[i][1], s=f"{img_chosen_lvl[i]}", fontsize=10)
            if img_scores is not None:
                plt.text(img_points[i][0], img_points[i][1], s=f"{(img_scores[i] * 100).round(2)}", color=(1, 1, 1),
                         fontsize=10)
        # plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=40, c='g')
        is_inside = inside(img_points, img_true_bboxes)
        plt.scatter(img_points[is_inside, 0], img_points[is_inside, 1], s=5, c='#ffff00', zorder=10)  # r
        plt.scatter(img_points[np.logical_not(is_inside), 0], img_points[np.logical_not(is_inside), 1], s=40,
                    c=(0, 0, 0))
        img_name = os.path.split(img_path)[-1]
        # plt.savefig("exp/debug/CPR/vis_2_{}".format(img_name))
        # plt.show()
        plt.clf()


if __name__ == '__main__':
    head = EcplHead(256, 80)
    print(head)
