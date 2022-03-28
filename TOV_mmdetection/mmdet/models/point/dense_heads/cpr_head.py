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

from mmcv.runner import BaseModule
from mmdet.models.losses import accuracy
from collections import defaultdict


def swap_list_order(alist):
    """
    Args:
        alist: shape=(B, num_level, ....)
    Returns:
        alist: shape=(num_level, B, ...)
    """
    new_order0 = len(alist[0])
    return [[alist[i][j] for i in range(len(alist))] for j in range(new_order0)]


def stack_feats(feats):
    """
    Args:
        feats: [B, (n, c)]
    Returns:
        feats: (B, N, c)
        valid: (B, N)
    """
    max_num = max([len(f) for f in feats])
    s0 = feats[0].shape
    for f in feats:
        assert f.shape[1:] == s0[1:], f"{f.shape} vs {s0}"

    shape = (max_num,) + feats[0].shape[1:]
    new_feats = []
    valids = []
    for feat in feats:
        new_feat = torch.zeros(shape, dtype=feat.dtype).to(feat.device)
        valid = torch.zeros(max_num, dtype=torch.bool).to(feat.device)
        new_feat[:len(feat)] = feat
        valid[: len(feat)] = 1
        new_feats.append(new_feat)
        valids.append(valid)
    return torch.stack(new_feats), torch.stack(valids)


def fill_list_to_tensor(alist, default_value=-1):
    max_l = max([len(l) for l in alist])
    data = torch.empty(len(alist), max_l, *alist[0].shape[1:]).type_as(alist[0]).to(alist[0])
    data[:] = default_value
    for i, l in enumerate(alist):
        data[i, :len(l)] = l
    return data


def group_by_label(data, labels):
    assert len(labels.shape) == 1
    labels = labels.cpu().numpy().tolist()
    label2data = defaultdict(list)
    for label, d in zip(labels, data):
        label2data[label].append(d)
    return {l: torch.stack(data) for l, data in label2data.items()}


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


def build_from_type(cfg, **kwargs):
    cls = cfg.pop('type')
    return eval(cls)(**cfg, **kwargs)


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


OutGridCirclesPtFeatGenerator = OutCirclePtFeatGenerator


class GridPtFeatGenerator(AnchorPtFeatGenerator):
    def __init__(self, max_pos_num=-1, *args, **kwargs):
        super(GridPtFeatGenerator, self).__init__(*args, **kwargs)
        self.max_pos_num = max_pos_num
    
    def generate(self, feats, img_meta, stride, fpn_lvl, centers, labels):
        """
        Args:
            feats: list[Tensor], (k, (1, C, H, W))
            img_meta: dict
            centers: (num_gts, num_refine, 2)
            labels: (num_gts, )
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            bag_feats: list[Tensor], [k, (num_centers, 1, num_chosen, C)]
            pts: (num_centers, 1, num_chosen, 3), 3 is (x, y, lvl)
            valid: (num_centers, 1, num_chosen, num_class)
        """
        # 1. get feature map and grid pts
        # pt_feats: list[Tensor], [k, (H, W, C)]
        # pts: (H, W, 3), 3 is (x, y, lvl)
        # valid: (H, W, 1)
        pt_feats, pts, valid = super().generate(feats, img_meta, stride, fpn_lvl, centers, labels)
        h, w, _ = valid.shape

        # 2. get chosen flag of each gt on grid map
        # chosens: (num_gt, H, W), bool, chosens[gt_i, y, x]=True means (x, y) is chosen for gt_i
        chosens, max_pos_num = self.get_chosen_neighbours(pts, centers, stride)

        # 3. get feature/ pts/ valid of chosen points
        num_gts, num_refine, _ = centers.shape
        max_pos_num += num_refine
        # print('max_pos_num', max_pos_num)
        pos_pts = torch.zeros(num_gts, max_pos_num, pts.shape[-1]).to(pts.device)
        bag_feats = [torch.zeros(num_gts, max_pos_num, pt_feat.shape[-1]).to(pt_feat.device)
                     for pt_feat in pt_feats]
        valid = torch.ones(num_gts, max_pos_num, 1).bool().to(pts.device)
        for i, chosen in enumerate(chosens):
            pos_pt_per_gt = pts[chosen].reshape(-1, pts.shape[-1])
            pos_pts[i, :len(pos_pt_per_gt)] = pos_pt_per_gt
            valid[i, len(pos_pt_per_gt):] = False

            for k, pt_feat in enumerate(pt_feats):
                pos_feat_per_gt = pt_feat[chosen].reshape(-1, pt_feat.shape[-1])
                bag_feats[k][i, :len(pos_feat_per_gt)] = pos_feat_per_gt

        # 4. get feats of center, insert at head of return results.
        centers_feats = self.extract_feat(feats, centers, stride)  # [k, (num_gts, num_refine, C)]
        for k, bag_feat in enumerate(bag_feats):
            bag_feats[k] = torch.cat([bag_feat, centers_feats[k].flip(dims=(1,))], dim=1).unsqueeze(dim=1)
        centers = self.append_other_info(centers, stride, valid)
        pos_pts = torch.cat([pos_pts, centers.flip(dims=(1,))], dim=1).unsqueeze(dim=1)
        centers_valid = torch.ones(num_gts, num_refine, 1).bool().to(valid.device)
        valid = torch.cat([valid, centers_valid.flip(dims=(1,))], dim=1).unsqueeze(dim=1)
        return bag_feats, pos_pts, valid

    def extract_feat(self, feats, pts, stride):
        """
        Args:
            feats: shape=[k, (1, C, H, W)]
            pts: shape=(..., num_chosen, 2)
            stride: float
        Returns:
            point_bag_feats: shape=(..., num_chosen, feat_channel)
        """
        sampled_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
        return sampled_feats
    
    def get_chosen_neighbours(self, pts, centers, stride):
        raise NotImplementedError


class GridEllipsePtFeatGenerator(GridPtFeatGenerator):
    def __init__(self, a_minus_c=-1, a_divide_c=-1, **kwargs):
        self.a_minus_c = a_minus_c  # a - c, while c is half of focal distance and a is half of long axis length
        self.a_divide_c = a_divide_c  # a / c
        assert a_divide_c < 0 or a_minus_c < 0, 'only one of a/c or a-c can set, while the unset one must give a' \
                                                ' negative value like -1.0 to represent it is invalid'
        assert a_divide_c >= 1 or a_minus_c >= 0
        super().__init__(**kwargs)

    def get_chosen_neighbours(self, pts, centers, stride):
        """
            get points in ellipse
            Args:
                centers: (num_gts, num_refine, 2)
            Returns:
        """
        H, W, _ = pts.shape
        num_gts, num_refine, _ = centers.shape
        assert centers.shape[1] == 2
        c = F.normalize(centers[:, 0] - centers[:, 1], p=2, dim=-1) / 2  # focal distance, (num_gts,)
        if self.a_minus_c > 0:
            a = self.a_minus_c + c
        elif self.a_divide_c > 0:
            a = self.a_divide_c * c
        else:
            raise ValueError

        # |p-f1| + |p-f2| <= 2a, for ellipse definition, distance sum of point to two focus point less
        # than 2*long axis length
        # (1, H, W, 1, 2) - (num_gt, 1, 1, num_refine, 2) => (num_gt, H, W, num_refine, 2) => normalize =>
        # (num_gt, H, W, num_refine) => sum => (num_gt, H, W)
        pts_xy = pts[..., :2]
        dis = pts_xy.reshape(1, H, W, 1, 2) - centers.reshape(num_gts, 1, 1, num_refine, 2)
        dis = torch.norm(dis, p=2, dim=-1).sum(dim=-1)
        chosens = dis <= 2 * a.reshape(-1, 1, 1) * stride  # (num_gt, H, W)
        return chosens, self.get_max_pos_num(a, c)

    def get_max_pos_num(self, a, c):
        if self.max_pos_num <= 0:
            b = (a * a - c * c) ** 0.5
            return 2 * a.ceil() * b.ceil()
        else:
            return self.max_pos_num


class GridCirclesPtFeatGenerator(GridPtFeatGenerator):
    def __init__(self, radius, **kwargs):
        self.radius = radius
        super().__init__(**kwargs)

    def get_chosen_neighbours(self, pts, centers, stride):
        """
            get points in grid circle
            Args:
                centers: (num_gts, num_refine, 2)
            Returns:
        """
        H, W, _ = pts.shape
        num_gts, num_refine, _ = centers.shape

        # (1, H, W, 1, 2) - (num_gt, 1, 1, num_refine, 2) => (num_gt, H, W, num_refine, 2) => normalize =>
        # (num_gt, H, W, num_refine)
        pts_xy = pts[..., :2]
        dis = torch.norm(pts_xy.reshape(1, H, W, 1, 2) - centers.reshape(num_gts, 1, 1, num_refine, 2), p=2, dim=-1)
        chosens = torch.any(dis <= self.radius * stride, dim=-1).squeeze(dim=-1)  # in any one of circles => ((num_gt, H, W)
        return chosens, self.get_max_pos_num()

    def get_max_pos_num(self):
        if self.max_pos_num <= 0:
            return 2 * (2*self.radius) ** 2
        else:
            return self.max_pos_num


class CirclePtFeatGenerator(PtFeatGenerator):
    def __init__(self, radius, start_angle=0, base_num_point=8, same_num_all_radius=False, append_center=True,
                 **kwargs):
        self.radius = radius
        self.start_angle = start_angle
        self.base_num_point = base_num_point
        self.same_num_all_radius = same_num_all_radius
        self.append_center = append_center
        super().__init__(**kwargs)
        self.depends = dict()

    def generate(self, feats, img_meta, stride, fpn_lvl, centers, labels):
        """
        Args:
            feats: list[Tensor], (k, (1, C, H, W))
            img_meta: dict
            centers: (num_gts, num_refine, 2)
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            bag_feats: list[Tensor], [k, (num_centers, num_refine, num_chosen, C)]
            pts: (num_centers, num_refine, num_chosen, 3), 3 is (x, y, lvl)
            valid: (num_centers, num_refine, num_chosen, num_class)
        """
        num_gts, num_refine, _ = centers.shape
        pts = self.get_point_neighbours(centers.flatten(0, 1), stride).reshape(num_gts, num_refine, -1, 2)
        valid = self.get_point_valid(pts, *img_meta['pad_shape'][:2])
        pts = self.append_other_info(pts, stride, valid)
        bag_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
        # valid_class = pts.new_zeros(*valid.shape, self.num_classes)  # repeat valid to num_class
        # valid_class[..., :] = valid[..., None]
        return bag_feats, pts, valid[..., None]

    def get_point_neighbours(self, centers, stride):
        """
        Args:
            centers: Tensor, shape=(num_gt_pts, 2), each point is record as (x, y)
            stride:
        Returns:
            choose points inside circle which radius is r * stride with gt_point as center.
            chosen_anchor_pts: shape=(num_gt_pts, num_chosen, 2)
        """
        assert len(centers.shape) == 2
        chosen_pts = []
        for i in range(self.radius):
            r = (i + 1) * stride
            num_pts = self.base_num_point if self.same_num_all_radius else (self.base_num_point * (i + 1))
            angles = torch.arange(num_pts).float().to(centers.device) / num_pts * 360 + self.start_angle
            angles = angles / 360 * np.pi * 2
            anchor_pts = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
            chosen_pts.append(anchor_pts)
        chosen_pts = torch.cat(chosen_pts).unsqueeze(dim=0) + centers.reshape(-1, 1, 2)

        # add gt to last one
        if self.append_center:
            chosen_pts = torch.cat([chosen_pts, centers.unsqueeze(dim=1)], dim=1)
        return chosen_pts

    def get_depend(self, stride, dis_r=1.5, return_pts=False):
        """
        chosen = [center]
        r_pts = pts[i:i+c]
        Returns:
        """
        pts = self.get_point_neighbours(torch.FloatTensor([[0, 0]]), stride)[0]
        depend = [[] for _ in range(len(pts))]
        idx = torch.arange(len(pts))
        in_circle_pts = pts[-1:]
        in_circle_idx = idx[-1:]
        s = 0
        for i in range(self.radius):
            num_pts = self.base_num_point if self.same_num_all_radius else (self.base_num_point * (i + 1))
            r_pts = pts[s: s + num_pts]
            r_idx = idx[s: s + num_pts]
            s += num_pts

            is_close = torch.cdist(r_pts[..., :2], in_circle_pts[..., :2], p=2) <= dis_r * stride
            for close, ri in zip(is_close, r_idx):
                depend[ri].extend(in_circle_idx[close].tolist())

            in_circle_pts = torch.cat([in_circle_pts, r_pts])
            in_circle_idx = torch.cat([in_circle_idx, r_idx])
        depend[-1].append(len(pts)-1)

        max_l = max([len(dep) for dep in depend])
        tensor_depend = torch.tensor([[dep[0]] * max_l for dep in depend])
        for ri, dep in enumerate(depend):
            assert len(dep) > 0
            tensor_depend[ri][:len(dep)] = torch.tensor(dep)
        if return_pts:
            return tensor_depend, pts
        else:
            return tensor_depend

    def filter_depend(self, stride, dis_r, valid):
        """
        Args:
            stride: float
            valid: (..., num_chosen)
        Returns:
        """
        if stride not in self.depends:
            assert isinstance(stride, (float, int)) and isinstance(dis_r, float)
            self.depends[(stride, dis_r)] = self.get_depend(stride, dis_r)
        depend = self.depends[(stride, dis_r)]
        for i, dep in enumerate(depend):
            valid[..., i] &= valid[..., dep].any(dim=-1)


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

    def forward(self, cls_feat, ins_feat, gt_r_points, gt_labels, img_metas, gt_points_ignore=None, ins_same_as_cls=True):
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


class PointRefiner(object):
    def __init__(self, strides, gt_alpha=0.5, merge_th=0.05, refine_th=0.05,
                 classify_filter=False, refine_pts_extractor=None, return_score_type='mean',
                 nearest_filter=True,
                 debug=False, debug_info=dict()):
        self.gt_alpha = gt_alpha
        self.merge_th = merge_th
        self.refine_th = refine_th
        self.strides = strides
        self.use_classify_filter = classify_filter
        self.use_nearest_filter = nearest_filter

        self.refine_pts_extractor = refine_pts_extractor
        # self.pos_generator = build_from_type(pos_generator)
        # self.other_generators = [build_from_type(g) for g in other_generators]
        self.return_score_type = return_score_type
        self.debug = debug

        TestCPRHead.DO_TEST = debug
        TestCPRHead.DBI['COUNT'] = debug_info.get('COUNT', -1)
        TestCPRHead.DBI['epoch'] = debug_info.get('epoch', -1)
        TestCPRHead.DBI['show'] = debug_info.get('show', True)

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
        return valid.reshape(num_gts, num_refine*num_chosen)

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
        bag_pts = bag_pts.reshape(num_gts, num_refine*num_chosen, -1)
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

        # 2. split gt prob out, last one in bag
        gt_cls_prob = bag_cls_prob[..., -1:, :]
        gt_r_pts = bag_pts[..., -1:, :]
        assert (gt_r_pts[:, :, 0, :2] == gt_r_points[:, :1]).all(), f"{gt_r_pts.shape} vs {gt_r_points.shape}"

        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        gt_idx = torch.arange(len(gt_labels))
        merge_valid = bag_valid.reshape(num_gts, num_refine*num_chosen).bool()
        # 3. assign point
        if self.use_nearest_filter:
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

        # shape: [len_gt, (K, 3)], got K-th
        chosen_pts = [bag_pts[gt_i][chosen] for gt_i, chosen in enumerate(bag_pts_weight > 0)]
        return refine_pts, refine_scores, not_refine, chosen_pts, self.get_geo_output(chosen_pts, refine_pts)

    def get_geo_output(self, chosen_pts, refine_pts):
        geos = []
        # for gt_i, pts in enumerate(chosen_pts):
        #     if len(pts) > 0:
        #         x, y = pts[:, 0], pts[:, 1]
        #         bbox = [x.min(), y.min(), x.max(), y.max()]
        #     else:
        #         x, y = refine_pts[gt_i, :2]
        #         bbox = [x, y, x, y]
        #     geos.append(bbox)
        for gt_i, (c_pts, r_pt) in enumerate(zip(chosen_pts, refine_pts)):
            geos.append(torch.cat([r_pt[None, :2], c_pts[:, :2]], dim=0))
        return geos

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

        refine_pts, refine_scores, not_refine, chosen_pts, refine_geos = multi_apply(
            self.refine_single, bag_data_list, grid_data_list, gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine)

        if self.debug:
            for i, img_meta in enumerate(img_metas):
                refine_pts, chosen_pts, gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine, refine_scores = \
                    refine_pts[i:i+1], chosen_pts[i:i+1], gt_r_points[i:i+1], gt_labels[i:i+1], img_metas[i:i+1],\
                    gt_true_bboxes[i:i+1], not_refine[i:i+1], refine_scores[i:i+1]
                TestCPRHead.get(self).test_refine_point(refine_pts, None, chosen_pts,
                                      gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine, refine_scores)
            # TestCPRHead.get(self).test_grid(grid_data_list, gt_labels, img_metas)
        return refine_pts, refine_scores, not_refine, refine_geos


@HEADS.register_module()
class CPRHead(AnchorFreeHead):
    """
    Coarse Point Refine Head
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_cls_fcs=0,
                 fc_out_channels=1024,
                 train_pts_extractor=dict(
                     pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                     neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3),
                 ),
                 refine_pts_extractor=dict(
                     pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                     neg_generator=dict(type='AnchorPtFeatGenerator', scale_factor=1.0),
                 ),
                 point_refiner=dict(
                 ),
                 ins_share_head_feat=True,
                 ins_share_head_classifier=False,
                 loss_mil=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=1.0),
                 loss_type=0,
                 loss_cfg=dict(
                     with_neg=True,
                     neg_loss_weight=1.0,
                     refine_bag_policy='independent_with_gt_bag',
                     random_remove_rate=0.4,
                     with_gt_loss=False,
                     gt_loss_weight=1.0,
                     with_mil_loss=True,
                 ),
                 normal_cfg=dict(
                     prob_cls_type='sigmoid',
                     out_bg_cls=False
                 ),
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_out',
                         std=0.01,
                         bias_prob=0.01),
                 ),
                 debug=False,
                 debug_info=dict(),
                 other_info=dict(),
                 **kwargs):
        self.num_cls_out = num_classes + 1 if normal_cfg["out_bg_cls"] else num_classes
        # if loss_mil['type'] == 'MIL2Loss':
        #     self.num_cls_out *= 2
        self.num_cls_fcs = num_cls_fcs
        self.fc_out_channels = fc_out_channels
        self.ins_share_head_feat = ins_share_head_feat
        self.ins_share_head_classifier = ins_share_head_classifier
        self.binary_ins = loss_mil['binary_ins']
        self.loss_cfg = loss_cfg

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_mil,
            init_cfg=init_cfg,
            **kwargs)

        s = self.strides
        self.train_pts_extractor = PointExtractor(**train_pts_extractor, strides=s, num_classes=num_classes)
        self.refine_pts_extractor = PointExtractor(**refine_pts_extractor, strides=s, num_classes=num_classes)
        self.point_refiner = PointRefiner(**point_refiner, debug=debug, debug_info=debug_info, strides=s,
                                          refine_pts_extractor=self.refine_pts_extractor)

        self.loss_mil = build_loss(loss_mil)
        self.loss_type = loss_type
        self.normal_cfg = normal_cfg

        self.debug = debug
        self.other_info = other_info

    def _init_layers(self):
        """Initialize layers of the head."""
        chn = self.in_channels
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.ins_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            if not self.ins_share_head_feat:
                self.ins_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            chn = self.feat_channels

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

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None,
                      proposal_cfg=None, **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_true_bboxes=gt_true_bboxes)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        if not self.ins_share_head_feat:
            ins_feat = x
            for ins_conv in self.ins_convs:
                ins_feat = ins_conv(ins_feat)
        else:
            ins_feat = cls_feat
        return cls_feat, ins_feat

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
            cls_outs, = multi_apply(get_outs_single, pts_cls_feats)
            return cls_outs
        else:
            cls_outs, ins_outs = multi_apply(get_outs_single, pts_cls_feats, pts_ins_feats)
            return cls_outs, ins_outs

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

    def loss(self, cls_feat, ins_feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, gt_true_bboxes=None, gt_weights=None):
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None
        gt_r_points = [pts.reshape(len(labels), -1, *pts.shape[1:]) for pts, labels in zip(gt_points, gt_labels)]
        # gt_r_points = [pts.unsqueeze(dim=1) for pts in gt_points]

        pos_data, neg_data = self.train_pts_extractor(
            cls_feat, ins_feat, gt_r_points, gt_labels, img_metas, gt_points_ignore, self.ins_share_head_feat)
        pos_data.cls_outs, pos_data.ins_outs = self.get_pts_outs(pos_data.cls_feats, pos_data.ins_feats)
        neg_data.cls_outs = self.get_pts_outs(neg_data.cls_feats)

        gt_labels_all = torch.cat(gt_labels, dim=0)
        gt_weights = [torch.FloatTensor([1.0] * len(l)) for l in gt_labels] if gt_weights is None else gt_weights
        gt_weights = torch.cat(gt_weights, dim=0).to(gt_labels_all.device)
        return getattr(self, f'loss{self.loss_type}')(pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes,
                                                      gt_weights)

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

    def loss0(self, pos_data: PtAndFeat, neg_data: PtAndFeat, gt_labels_all, gt_r_points,
              gt_true_bboxes=None, gt_weights=None):
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
            gt_cls_outs = pos_cls_outs[..., -1, :].reshape(num_gts*num_refine, -1)
            gt_cls_prob = self.get_cls_prob(gt_cls_outs)

            gt_loss_type = self.loss_cfg.get('gt_loss_type', 'gt_refine')
            if gt_loss_type == 'mil':
                raise NotImplementedError
            else:
                if gt_loss_type == 'gt_refine':
                    gt_labels_rep = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_valid = pos_valid[..., -1, :].reshape(num_gts*num_refine, -1)
                    gt_weights_rep = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_weights_rep = gt_valid.float() * gt_weights_rep.reshape(-1, 1)
                elif gt_loss_type == 'gt':
                    gt_labels_rep = gt_labels_all
                    gt_weights_rep = pos_valid[:, 0, -1, :].float() * gt_weights.reshape(-1, 1)
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
                pos_pts = pos_pts[:, si:].reshape(num_gts, (num_refine-si) * num_chosen, 3)
                pos_cls_outs = pos_cls_outs[:, si:].reshape(num_gts, (num_refine-si) * num_chosen, -1)
                pos_ins_outs = pos_ins_outs[:, si:].reshape(num_gts, (num_refine-si) * num_chosen, -1)
                pos_valid = pos_valid[:, si:].reshape(num_gts, (num_refine-si) * num_chosen, -1)
                pos_weights = gt_weights
            else:
                raise ValueError
            pos_weights = pos_valid.float() * pos_weights.reshape(-1, 1, 1)

            self.random_remove(pos_pts, neg_pts, random_remove_rate=self.loss_cfg['random_remove_rate'])

            pos_cls_prob = self.get_cls_prob(pos_cls_outs)
            pos_loss, bag_acc, num_pos = self.loss_mil(pos_cls_prob, pos_ins_outs, gt_labels_all, pos_weights)
            losses.update({"pos_loss": pos_loss, "bag_acc": bag_acc})

        if self.loss_cfg.get("with_neg", True):
            neg_prob = self.get_cls_prob(neg_cls_outs)
            num_neg, num_class = neg_prob.shape
            neg_labels = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_prob.device)
            loss_weights = self.loss_cfg["neg_loss_weight"]
            neg_valid = neg_valid.reshape(num_neg, -1)

            neg_loss = self.loss_mil.gfocal_loss(neg_prob, neg_labels, neg_valid.float())
            neg_loss = loss_weights * weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
            losses.update({"neg_loss": neg_loss})
        return losses

    def get_bboxes(self, cls_feat, ins_feat, img_metas,
                   cfg=None, rescale=False, with_nms=True,
                   gt_bboxes=None, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None, gt_anns_id=None,
                   not_refine=None, cascade_out_fmt=False):
        """
        Args:
            gt_bboxes: [num_img, (num_gts*num_refine, 4)]
            gt_labels: [num_img, (num_gts,)]
            gt_bboxes_ignore:
            gt_true_bboxes: [num_img, (num_gts, 4)] or None
        """
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None
        gt_r_points = [pts.reshape(len(labels), -1, *pts.shape[1:]) for pts, labels in zip(gt_points, gt_labels)]
        # gt_r_points = [pts.unsqueeze(dim=1) for pts in gt_points]

        bag_data, grid_data = self.refine_pts_extractor(
            cls_feat, ins_feat, gt_r_points, gt_labels, img_metas, gt_points_ignore, self.ins_share_head_feat)
        bag_data.cls_outs, bag_data.ins_outs = self.get_pts_outs(bag_data.cls_feats, bag_data.ins_feats)
        grid_data.cls_outs = self.get_pts_outs(grid_data.cls_feats)

        bag_data.cls_prob = [self.get_cls_prob(bag_cls_feat_lvl) for bag_cls_feat_lvl in bag_data.cls_outs]
        grid_data.cls_prob = [self.get_cls_prob(grid_cls_outs_lvl) for grid_cls_outs_lvl in grid_data.cls_outs]

        points, scores, not_refine, geos = self.point_refiner(bag_data, grid_data, gt_r_points, gt_labels, img_metas,
                                                        gt_true_bboxes, not_refine)

        assert sum([len(l) for l in gt_labels]) == sum([len(p) for p in points])
        final_det_bboxes = []
        det_bboxes = self.center_to_pseudo_bbox(points)
        for im_id, bboxes in enumerate(det_bboxes):
            geos_im = geos[im_id]
            if rescale:
                scale_factor = img_metas[im_id]['scale_factor']
                bboxes /= bboxes.new_tensor(scale_factor)
                geos_im = self.scale_geos(geos_im, scale_factor)  # [num_gt]
            scores_img = scores[im_id].unsqueeze(dim=-1)
            anns_id = gt_anns_id[im_id].unsqueeze(dim=-1).type_as(scores_img)
            geos_im = fill_list_to_tensor(geos_im)
            geos_im = geos_im.reshape(len(geos_im), -1)
            if self.other_info.get('out_geo', False):
                final_det_bboxes.append(torch.cat((bboxes, scores_img, anns_id, geos_im), dim=-1))
            else:
                final_det_bboxes.append(torch.cat((bboxes, scores_img, anns_id), dim=-1))

        if cascade_out_fmt:
            return list(zip(final_det_bboxes, gt_labels)), not_refine

        if with_nms:
            return list(zip(final_det_bboxes, gt_labels))
        else:
            raise NotImplementedError

    def get_targets(self):
        pass

    def scale_geos(self, geos_im, scale_factor):
        for gt_id, geos_gt in enumerate(geos_im):
                geos_gt[:] /= geos_gt.new_tensor(scale_factor[:2])
        return geos_im

    def pseudo_bbox_to_center(self, gt_bboxes):
        """
        Transform pseudo bbox to center point
        Args:
            gt_bboxes: [num_imgs, (num_pts, 2)]
        Returns:
        """
        return [(gt_bboxes_img[:, :2] + gt_bboxes_img[:, 2:]) / 2
                for gt_bboxes_img in gt_bboxes]

    def center_to_pseudo_bbox(self, centers, pseudo_wh=(16, 16)):
        """
        Returns:
        """
        if not isinstance(pseudo_wh, torch.Tensor):
            pseudo_wh = centers[0].new_tensor(pseudo_wh)
        return [torch.cat([center - pseudo_wh / 2, center + pseudo_wh / 2], dim=-1) for center in centers]


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
    DO_TEST = False
    DBI = dict(epoch=12, COUNT=-1, show=True)

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
                plt.subplot(2, k, i+1)
                plt.imshow(mask_img(grid_cls_prob[:, :, l], img))
                max_score = grid_cls_prob[:, :, l].max().round(2)
                plt.title(f"pos: {l}({TestCPRHead.CLASSES[l]}); max_score: {str(max_score)}")
            for i, l in enumerate(neg_labels[:k]):
                plt.subplot(2, k, k+i+1)
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

    def save_cpr_data(self, img_meta, **kwargs):
        img_path = img_meta['filename']
        import os
        if not os.path.exists(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/'):
            os.makedirs(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/')
        img_name = os.path.split(img_path)[-1]
        np.savez(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/' + img_name + '.npz', dict=kwargs)

    def test_refine_point(self, points, chosen_lvl, chosen_pts_all, gt_r_points, gt_labels, img_metas, gt_true_bboxes,
                          not_refine, fmt_points_score):
        if not TestCPRHead.DO_TEST:
            return
        self.count += 1
        # print(TestCPRHead.DBI['COUNT'])
        if 0 <= TestCPRHead.DBI['COUNT'] < self.count:
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

        def plot_img(img_meta):
            img_path = img_meta['filename']
            sw, sh = img_meta['scale_factor'][:2]

            img = Image.open(img_path)
            if 'corner' in img_meta:
                img = img.crop(img_meta['corner'])
            w, h = img.width, img.height
            print(w, h)
            img = img.resize((round(int(w * sw)), int(round(h * sh))))
            img = np.array(img)
            plt.figure(figsize=(14, 8))
            fig = set_plt(plt)
            plt.imshow(img)

        def plot_points(annotated_points, chosen_points, refined_points, category_anns, img_true_bboxes=None):
            # colors = get_hsv_colors(80)
            colors = [(0.65, 0.65, 0.65)] * 80

            if chosen_points is not None:
                for i in range(len(annotated_points)):
                    # link ann_pt -- chosen_pts
                    for j in range(len(chosen_points[i])):
                        p1, p2 = annotated_points[i], chosen_points[i][j]
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], ':', color=colors[category_anns[i]])  #
                    # link ann_pt -- refine_pt
                    plt.plot([annotated_points[i, 0], refined_points[i, 0]], [annotated_points[i, 1], refined_points[i, 1]],
                             '--', linewidth=3, color=colors[category_anns[i]])
                    plt.scatter(chosen_points[i][:, 0], chosen_points[i][:, 1], s=4, c='#fa2338', zorder=11)  #

            # 1. ann_pt
            # plt.scatter(annotated_points[:, 0], annotated_points[:, 1], s=120, c='#22fe61', zorder=11)
            plt.scatter(annotated_points[:, 0], annotated_points[:, 1], s=12, c='#22fe61', zorder=11)
            # refined_pt
            # plt.scatter(refined_points[:, 0], refined_points[:, 1], s=120, c='#ffff00', zorder=11)  # yellow
            plt.scatter(refined_points[:, 0], refined_points[:, 1], s=12, c='#ffff00', zorder=11)  # yellow

            # is_inside = inside(refined_points, img_true_bboxes)
            # plt.scatter(refined_points[is_inside, 0], refined_points[is_inside, 1], s=120, c='#ffff00', zorder=11) # yellow
            # plt.scatter(refined_points[np.logical_not(is_inside), 0], refined_points[np.logical_not(is_inside), 1],
            #             s=40, c=(0, 0, 0))

        def plot_scores(refined_points, img_scores):
            if img_scores is not None:
                for i in range(len(refined_points)):
                    plt.text(refined_points[i][0], refined_points[i][1], s=f"{(img_scores[i] * 100).round(2)}",
                             color=(1, 1, 1), fontsize=10)

        self.test_refine_point2(points, gt_true_bboxes, not_refine)

        img_path = img_metas[0]['filename']
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

        if not os.path.exists("exp/debug/CPR"):
            os.makedirs("exp/debug/CPR")

        annotated_points, chosen_points, refined_points = img_gt_points, img_chosen_pts, img_points

        print(img_metas)

        plot_img(img_metas[0])
        plot_points(annotated_points, chosen_points, refined_points, img_gt_labels, img_true_bboxes)
        # plot_scores(refined_points, img_scores)
        img_name = os.path.split(img_path)[-1]
        plt.savefig("exp/debug/CPR/vis_{}".format(img_name))
        if TestCPRHead.DBI['show']:
            plt.show()
        else:
            plt.clf()

        plot_img(img_metas[0])
        plot_points(annotated_points, None, refined_points, img_gt_labels, img_true_bboxes)
        # plot_scores(refined_points, img_scores)
        img_name = os.path.split(img_path)[-1]
        plt.savefig("exp/debug/CPR/vis2_{}".format(img_name))
        if TestCPRHead.DBI['show']:
            plt.show()
        else:
            plt.clf()


if __name__ == '__main__':
    # [-0.5, w-1+0.5] -> [-1, 1]
    # x -> x' => x' = (2x+1) / w - 1
    grid_map_func = lambda xy, wh: (2 * xy + 1) / wh - 1
    input = torch.arange(16).reshape(1, 1, 4, 4).float()
    # grid = torch.tensor([[[
    #     [-1, -1], [-0.5, -0.5], [0, 0], [0.5, 0.5], [1, 1]
    # ]]]).float()
    grid = torch.tensor([[[
        [0, 0], [1, 1], [2, 2]
    ]]]).float()
    grid = grid_map_func(grid, grid.new_tensor([input.shape[-1], input.shape[-2]]))
    x = F.grid_sample(input, grid, padding_mode='border', align_corners=False)
    print(input)
    print(x)

    # [0, w-1] -> [-1, 1]
    grid_map_func = lambda xy, wh: 2 * xy / (wh - 1) - 1
    grid = torch.tensor([[[
        [0, 0], [1, 1], [2, 2]
    ]]]).float()
    grid = grid_map_func(grid, grid.new_tensor([input.shape[-1], input.shape[-2]]))
    x = F.grid_sample(input, grid, padding_mode='border', align_corners=True)
    print(input)
    print(x)

    from PIL import Image
    import matplotlib.pyplot as plt

    # k line: X[:, k], Y[:, k]
    plt.plot([[0, 1], [2, 3]], [[8, 9], [10, 4]])
    plt.plot()
    plt.show()

    # print(torch.topk(torch.tensor([[1, 2, 30, 9, 7, 6], [1, 2, 30, 9, 7, 10]]), 3, largest=False, dim=1))
    #
    # print(torch.tensor([[1, 2, 3], [4, 5, 6]]).float().norm(dim=-1))
