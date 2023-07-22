import torch
import numpy as np
import torch.nn.functional as F
from .utils import group_by_label, groups_by_label, grid_sample, MultiList, Input, PtAndFeat, \
    sample_by_flag, swap_list_order, Statistic
from mmcv.runner import BaseModule
from copy import deepcopy
from mmdet.utils.logger import get_root_logger
from ssdcv.dbg.logger import logger


def build_from_type(cfg, **kwargs):
    cls = cfg.pop('type')
    return eval(cls)(**cfg, **kwargs)


def xyxy2centerwh(gt_bboxes):
    """
    Transform pseudo bbox to center point
    Args:
        gt_bboxes: (num_bboxes, 4)
    Returns:
    """
    return torch.cat([(gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2,
                      (gt_bboxes[:, 2:] - gt_bboxes[:, :2])], dim=-1)


def get_wh_tuple(half_wh):
    if isinstance(half_wh, (int, float)):
        print(f"get_wh from (int, float) {half_wh}")
        half_wh = (half_wh, half_wh)
    elif isinstance(half_wh, (tuple, list)):
        print(f"get_wh from (tuple, list) {half_wh}")
    elif isinstance(half_wh, str):
        print(f"get_wh from str {half_wh}")
        half_wh = eval(half_wh)
    else:
        raise TypeError(f"{type(half_wh)} {half_wh}")
    return half_wh


class SampleFeatExtractor(BaseModule):
    """generate pos bag and neg points and extract them features"""

    def __init__(self, strides: tuple, num_classes,
                 pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                 neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3),
                 debug_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.test_sampler = TestSampler(debug_cfg=debug_cfg)
        self.pos_generator = build_from_type(pos_generator, num_classes=num_classes, test_sampler=self.test_sampler)
        self.neg_generator = build_from_type(neg_generator, num_classes=num_classes, test_sampler=self.test_sampler)
        self.strides = strides
        self.debug_cfg = debug_cfg

    def forward(self, cls_feat, ins_feat, input_data: Input, ins_same_as_cls=True):
        """
        Returns:
            pos_data.cls_feats: [num_lvl, (num_gts, num_refine, num_chosen, C)]
            neg_data.cls_feats: [num_lvl, (num_negs, C)]
        """
        if ins_same_as_cls:  # if ins_feat same as cls_feat, extract only one is enough.
            (pos_cls_feat,), pos_data, (neg_cls_feat,), neg_data = self.extract(
                (cls_feat,), input_data)
            pos_ins_feat = pos_cls_feat
        else:
            (pos_cls_feat, pos_ins_feat), pos_data, (neg_cls_feat, _), neg_data = self.extract(
                (cls_feat, ins_feat), input_data)
        pos_data.cls_feats, pos_data.ins_feats = pos_cls_feat, pos_ins_feat
        neg_data.cls_feats = neg_cls_feat
        return pos_data, neg_data

    def extract(self, all_feats, input_data: Input):
        """
        Returns:
            pos_feats: [k, num_lvl, (num_gt_all_img, num_refine, num_chosen, C)]
            pos_pts:   [num_lvl, (num_gt_all_img, num_refine, num_chosen, 4)]
            neg_feats: [k, num_lvl, (num_neg_all_img, C)]
            neg_pts:   [num_lvl, (num_neg_all_img, 4)]
        """
        pos_feats, pos_pts, pos_valid = self.pos_generator(self.strides, all_feats, input_data)
        neg_feats, neg_pts, neg_valid = self.neg_generator(self.strides, all_feats, input_data)

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


class FPNLevelAssigner(object):
    def __init__(self, *args, **kwargs):
        self.statistic_obj = Statistic()

    def __call__(self, input_data: Input, strides):
        """
        return:
            chosen_lvl_all: [B, (num_gt,)]
        """
        raise NotImplementedError

    def statistic(self, chosen_lvl_all, num_lvl):
        chosen_lvl_all = torch.cat(chosen_lvl_all, dim=0)
        lvl_count = [0] * num_lvl
        for lvl in chosen_lvl_all:
            lvl_count[lvl] += 1
        lvl_count = chosen_lvl_all.new_tensor(lvl_count).unsqueeze(dim=0)
        self.statistic_obj.print_mean("gt count on each level of fpn", lvl_count, log_interval=1000)


class FirstLvlAssigner(FPNLevelAssigner):
    def __call__(self, input_data: Input, strides):
        chosen_lvl_all = []
        for img_i in range(len(input_data)):
            gt_labels = input_data.get(img_i).gt_labels
            chosen_lvl_all.append(torch.zeros(len(gt_labels)).long().to(gt_labels))  # chosen first_lvl
        input_data.chosen_lvl = chosen_lvl_all


class SizeAssigner(FPNLevelAssigner):
    def __init__(self, base_scale=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_scale = base_scale

    def __call__(self, input_data: Input, strides):
        chosen_lvl_all = []
        for img_i in range(len(input_data)):
            # 1. get size of predict bbox
            input_data_img = input_data.get(img_i)
            gt_r_pts, labels, refined_bbox = input_data_img.gt_r_pts, input_data_img.gt_labels, \
                                             input_data_img.refined_geos
            assert refined_bbox.shape[1] == 4, "refine_geos must be 4D bbox."
            pred_bbox = xyxy2centerwh(refined_bbox)
            centers = pred_bbox[:, :2].unsqueeze(dim=1)
            size = (pred_bbox[:, 2] * pred_bbox[:, 3]) ** 0.5

            # 2. assign to FPN lvl wrt size
            # log_stride = torch.log2(torch.tensor(strides))
            chosen_lvl = torch.log2(size / self.base_scale / strides[0])
            chosen_lvl = torch.clamp(chosen_lvl, min=0, max=len(strides) - 1).long()
            chosen_lvl_all.append(chosen_lvl)
            logger.max_log("max pseudo bbox size", size.max().cpu().numpy(), log_func=get_root_logger().info)
        input_data.chosen_lvl = chosen_lvl_all
        if len(strides) > 1:
            self.statistic(chosen_lvl_all, len(strides))
        # return chosen_lvl_all  # [B, (num_gt)]


class BaseGenerator(object):
    """
        1. dynamic size handler
        2. tool function: assert function; inside image valid; extract point feat; append stride info to pts;
    """

    def __init__(self, size, num_classes, align_corners=False, test_sampler=None, is_pos=True,
                 dynamic_size_alpha=-1.0, dynamic_default_size=8, dynamic_min_size=1,
                 dynamic_with_input=False, dynamic_base_input_size=400,
                 radius=None, half_wh=None,  # useless, just for pass args
                 ):
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.test_sampler = test_sampler
        self.dynamic_with_input = dynamic_with_input
        self.dynamic_base_input_size = dynamic_base_input_size
        self.set_use_dynamic_size(size, dynamic_size_alpha, dynamic_default_size, dynamic_min_size)

        self.is_pos = is_pos  # is positive sample generator

    # tool function
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
        # cause feat map range from [0, w], while grid_sample range from [-0.5, w-0.5];
        chosen_pts = chosen_pts - 0.5
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

    # for dynamic generator:
    def get_center_size(self, input_image_data, stride):
        """

        return:
            centers: (num_gt, num_refine=1, 2)
            size: (num_gt, ) / (num_gt, k), such as radius/half_wh
        """
        raise ValueError("Not Implement.")

    def set_use_dynamic_size(self, size, dynamic_size_alpha, dynamic_default_size, dynamic_min_size):
        """
            size: int, tuple(int)
        """
        if self.dynamic_with_input:
            self.use_dynamic_size = True
            return

        size = torch.tensor(size)
        if torch.all(size >= 0):  # size > 0
            assert dynamic_size_alpha < 0, "size have been set, can not using use dynamic size, " \
                                           "please set size=-1 or dynamic_size_alpha=-1" + \
                                           f", but got {size} and {dynamic_size_alpha}"
            self.use_dynamic_size = False
        else:
            assert dynamic_size_alpha >= 0, "size and dynamic_size_alpha have not been set " \
                                            "please set size>=0 or dynamic_size_alpha>=0" + \
                                            f", but got {size} and {dynamic_size_alpha}"
            self.use_dynamic_size = True
            self.dynamic_size_alpha = dynamic_size_alpha
            self.dynamic_default_size = dynamic_default_size
            self.dynamic_min_size = dynamic_min_size

    def set_max_size_of_batch(self, input_data: Input, strides):
        if hasattr(self, "use_dynamic_size") and getattr(self, "use_dynamic_size"):
            # assert len(strides) == 1
            # stride = strides[0]
            sizes = []
            for img_id in range(len(input_data)):
                input_data_img = input_data.get(img_id)
                stride = strides[input_data_img.chosen_lvl]
                centers, sizes_img = self.get_center_size(input_data_img, stride)
                sizes.append(sizes_img)
            max_size = torch.cat(sizes, dim=0).max(dim=0)[0]
            input_data.other_data = dict(max_size_in_batch=max_size)
            logger.max_log("max_size", max_size, log_func=get_root_logger().info)

    # for dynamic radius generator, Implement of get_center_size for circle generator
    def get_center_radius(self, input_data_img, stride):
        """
        stride
            centers: (num_gt, num_refine=1, 2)
            radius: (num_gt, )
        """
        gt_r_pts, labels, refined_bbox = input_data_img.gt_r_pts, input_data_img.gt_labels, input_data_img.refined_geos
        if not self.use_dynamic_size:
            radius = labels.new_tensor([self.radius] * len(labels))
            return gt_r_pts, radius
        else:
            if self.dynamic_with_input:
                # give the radius from input args
                radius = labels.new_tensor([self.radius] * len(labels))
                h, w, c = input_data_img.img_metas['img_shape']
                radius = radius.float() * (min(h, w) * 1.0 / self.dynamic_base_input_size)  # base_on 400
                radius = radius.round().type_as(labels)
                return gt_r_pts, radius

            # cal radius from refined box
            assert refined_bbox.shape[1] == 4, "refine_geos must be 4D bbox."
            pred_bbox = xyxy2centerwh(refined_bbox)
            centers = pred_bbox[:, :2].unsqueeze(dim=1)
            size = (pred_bbox[:, 2] * pred_bbox[:, 3]) ** 0.5
            radius = size * self.dynamic_size_alpha / 2 / stride
            illegal = radius < self.dynamic_min_size
            centers[illegal] = gt_r_pts[illegal]
            radius[illegal] = self.dynamic_default_size
            if not torch.all(radius > 0):
                print()
            assert torch.all(radius > 0), f"{radius} {self.dynamic_default_size}"
            return centers, radius.ceil().long()

    def get_center_half_wh(self, input_data_img, stride):
        """
            centers: (num_gt, num_refine=1, 2)
            half_wh: (num_gt, 2)
        """
        gt_r_pts, labels, refined_bbox = input_data_img.gt_r_pts, input_data_img.gt_labels, input_data_img.refined_geos
        if not self.use_dynamic_size:
            return gt_r_pts, labels.new_tensor([self.half_wh] * len(labels))
        else:
            assert refined_bbox.shape[1] == 4, "refine_geos must be 4D bbox."
            pred_bbox = xyxy2centerwh(refined_bbox)
            centers = pred_bbox[:, :2].unsqueeze(dim=1)
            half_wh = pred_bbox[:, 2:4] / 2
            shape = [-1] + ([1] * (len(half_wh.shape) - 1))
            half_wh = half_wh * self.dynamic_size_alpha / stride.reshape(*shape)
            illegal = (half_wh < self.dynamic_min_size).any(dim=-1)
            centers[illegal] = gt_r_pts[illegal]
            half_wh[illegal] = self.dynamic_default_size
            return centers, half_wh.ceil().long()


class PtFeatGenerator(BaseGenerator):
    """
    handle multi-level FPN
    """

    def __init__(self, size, num_classes,
                 only_first_lvl=True, fpn_level_assigner=dict(type='FirstLvlAssigner'),
                 *args, **kwargs):
        super().__init__(size, num_classes, *args, **kwargs)
        self.only_first_lvl = only_first_lvl
        self.fpn_level_assigner = build_from_type(deepcopy(fpn_level_assigner)) \
            if fpn_level_assigner is not None else None

        if self.fpn_level_assigner is None:
            assert not self.only_first_lvl, "must use all level while fpn_level_assigner is not given."

    def generate(self, feats, stride, lvl, input_data: Input):
        """generate points and them features for single FPN level of single image """
        raise NotImplementedError()

    def __call__(self, strides, all_feats, input_data: Input):
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
        assert isinstance(all_feats, (tuple, list))
        strides = input_data.gt_pts[0].new_tensor(strides)  # to gpu and float
        k = len(all_feats)

        if self.fpn_level_assigner is None:
            all_res = self.extract_feat_on_all_lvl(strides, all_feats, input_data)
        else:
            all_res = self.extract_feat_on_assigned_lvl(strides, all_feats, input_data)

        for i, data in enumerate(all_res.data):  # [k, B, num_lvl, (..., C)] => [k, num_lvl, B, (..., C)]
            all_res.data[i] = swap_list_order(all_res.data[i])

        feats, pts, valid = all_res.data[:k], all_res.data[k], all_res.data[k + 1]
        return feats, pts, valid

    def extract_feat_on_all_lvl(self, strides, all_feats, input_data: Input):
        """
            each gt chosen all lvl to extract feat
        """
        all_res = MultiList()  # [k, B, num_lvl, (..., C)]
        for img_id in range(len(input_data)):
            outs_img = MultiList()  # [k, num_lvl, (..., C)]
            input_data_img = input_data.get(img_id)
            for lvl, stride in enumerate(strides):
                ori_feats = [f[lvl][img_id:img_id + 1] for f in all_feats]
                feats_lvl, pts_lvl, valid_lvl = self.generate(ori_feats, stride, lvl, input_data_img)
                outs_img.append(*feats_lvl, pts_lvl, valid_lvl)
                if lvl == 0 and img_id == 0:
                    self.test_sampler.show(ori_feats, feats_lvl, pts_lvl, valid_lvl, input_data_img)
            # feats, pts, valid = outs_img.data[:k], outs_img.data[k], outs_img.data[k+1]
            all_res.append(*outs_img.data)
        return all_res

    def extract_feat_on_assigned_lvl(self, strides, all_feats, input_data: Input):
        """
            each gt only chosen single feat lvl to extract feat
        """
        if self.only_first_lvl:  # only use the lowest feat.
            strides = strides[:1]
            all_feats = [feats[:1] for feats in all_feats]
        self.fpn_level_assigner(input_data, strides)

        # set some args that all image in same batch shared.
        self.set_max_size_of_batch(input_data, strides)
        k = len(all_feats)

        all_res = MultiList()  # [k, B, num_lvl, (..., C)]
        for img_id in range(len(input_data)):
            outs_img = MultiList()  # [k, num_lvl, (..., C)]
            for lvl, stride in enumerate(strides):
                ori_feats = [f[lvl][img_id:img_id + 1] for f in all_feats]
                input_data_img = input_data.get(img_id, lvl)
                if self.is_pos:
                    if len(input_data_img.gt_labels) > 0:
                        feats_lvl, pts_lvl, valid_lvl = self.generate(ori_feats, stride, lvl, input_data_img)
                        outs_img.data = input_data.set(img_id, lvl, feats_lvl + [pts_lvl, valid_lvl], outs_img.data)
                    else:
                        continue
                else:
                    feats_lvl, pts_lvl, valid_lvl = self.generate(ori_feats, stride, lvl, input_data_img)
                    outs_img.append(*feats_lvl, pts_lvl, valid_lvl)
                if lvl == 0 and img_id == 0:
                    self.test_sampler.show(ori_feats, feats_lvl, pts_lvl, valid_lvl, input_data_img)
            # feats, pts, valid = outs_img.data[:k], outs_img.data[k], outs_img.data[k+1]
            all_res.append(*outs_img.data)
        return all_res


class AnchorPtFeatGenerator(PtFeatGenerator):
    def __init__(self, size, scale_factor=None, **kwargs):
        """
        Args:
            scale_factor: rescale of the feature when sample point from feature map
        """
        self.scale_factor = scale_factor
        super().__init__(size=size, **kwargs)

    def generate(self, feats, stride, lvl, input_data_img: Input):
        """
        Args:
            feats: list[Tensor], [k, (1, C, H, W)]
            img_meta: dict
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            pt_feats: list[Tensor], [k, (H, W, C)]
            pts: (H, W, 3), 3 is (x, y, lvl)
            valid: (H, W, 1)
            stride: fpn_stride / self.scale_factor
        """
        img_meta = input_data_img.img_metas
        if self.scale_factor and self.scale_factor != 1.0:
            feats = [F.interpolate(feat, scale_factor=self.scale_factor, mode='bilinear',
                                   align_corners=self.align_corners, recompute_scale_factor=False) for feat in feats]
            stride = stride / self.scale_factor

        h, w = self.assert_same_size(feats)
        pts, valid = self.anchor_points(h, w, *img_meta['pad_shape'][:2], stride, feats[0].device)
        pts = self.append_other_info(pts, stride, valid)
        pt_feats = [feat.permute(0, 2, 3, 1).squeeze(0) for feat in feats]
        # valid_class = pts.new_zeros(*valid.shape, self.num_classes)  # repeat valid to num_class
        # valid_class[..., :] = valid[..., None]
        return pt_feats, pts, valid[..., None], stride

    def anchor_points(self, h, w, valid_h, valid_w, stride, device):
        x, y = torch.arange(w).to(device), torch.arange(h).to(device)
        y, x = torch.meshgrid(y, x)
        pts = torch.stack([x, y], dim=-1) * stride + stride / 2
        return pts, self.get_point_valid(pts, valid_h, valid_w)


class OutPtFeatGenerator(AnchorPtFeatGenerator):
    def __init__(self, size, class_wise=False, keep_wh=False, near_gt_r=1.0, **kwargs):
        self.class_wise = class_wise
        self.keep_wh = keep_wh
        self.near_gt_r = near_gt_r

        super().__init__(size=size, is_pos=False, **kwargs)

    def generate(self, feats, stride, fpn_lvl, input_data_img: Input):
        """
        Args:
            feats: list[Tensor], (k, (1, C, H, W))
            img_meta: dict
            centers: (num_gts, num_refine, 2)
            labels: (num_gts, )
            refined_geos: (num_gts, 4), 4 is (x1, y1, x2, y2)
            chosen_lvl: (num_gts, )
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            pt_feats: list[Tensor], [k, (num, C)]
            pts: (num, 3), 3 is (x, y, lvl)
            valid: (num, num_class)  # (H, W, num_class)
        """
        gt_r_pts, labels, refined_geos = input_data_img.gt_r_pts, input_data_img.gt_labels, input_data_img.refined_geos

        pt_feats, pts, valid, stride = super().generate(feats, stride, fpn_lvl, input_data_img)
        h, w, _ = valid.shape
        pts, valid = pts.flatten(0, -2), valid.flatten(0, -2)

        repeat_l = [1] * len(valid.shape[:-1]) + [self.num_classes]
        valid = valid.repeat(*repeat_l)

        if len(labels) > 0:  # num_gt > 0
            centers, sizes = self.get_center_size(input_data_img, stride)
            if self.class_wise:
                label2centers, label2gts, label2sizes = groups_by_label((centers, gt_r_pts, sizes), labels)
                for label in label2centers:
                    chosen = self.get_chosen(pts, label2centers[label], label2sizes[label], label2gts[label], stride)
                    valid[..., label] = valid[..., label].float() * chosen.float()
            else:
                chosen = self.get_chosen(pts, centers, sizes, gt_r_pts, stride)
                valid = valid.float() * chosen[..., None].float()

        if self.keep_wh:
            valid = valid.reshape(h, w, -1)
            pts = pts.reshape(h, w, -1)
        else:
            pt_feats = [feat.flatten(0, -2) for feat in pt_feats]
        return pt_feats, pts, valid.bool()

    def get_chosen(self, pts, centers, sizes, gt_r_pts, stride):
        """
        args:
            pts: (num_pts=H*W, 3)
            gt_r_pts: (num_gt, num_refine, 2)
            centers: (num_gt, 1/num_refine, 2)
            sizes: (num_gt, ) / (num_gt, 2), radius/wh
        Return:
            chosen: (H*W, )
        """
        raise NotImplementedError


class OutCirclePtFeatGenerator(OutPtFeatGenerator):
    def __init__(self, radius, **kwargs):
        self.radius = radius
        super().__init__(radius, **kwargs)
        self.get_center_size = self.get_center_radius

    def get_chosen(self, pts, centers, radius, gt_r_pts, stride):
        """
            pts: (num_pts=H*W, 3)
            gt_r_pts: (num_gt, num_refine, 2)
            centers: (num_gt, 1/num_refine, 2)
            radius: (num_gt, )
        """
        # 1. we define anchor point in circle(center=gt,radius=stride*dynamic_near_gt_r) as near gt points,
        # which can't be negative
        gts = gt_r_pts.flatten(0, 1)  # (num_gt*num_refine, 2)
        dist = torch.cdist(pts[..., :2], gts, p=2)  # (H*W, num_gt*num_refine)
        not_near_gts = dist.min(dim=1)[0] >= stride * self.near_gt_r
        # 2. outside circle(center, radius) can be negative
        centers = centers.flatten(0, 1)  # (num_gt * num_refine, 2)
        dist = torch.cdist(pts[..., :2], centers, p=2)  # (H*W, num_gt*num_refine)
        chosen = torch.all(dist >= stride * radius, dim=1) & not_near_gts
        return chosen


class OutRectPtFeatGenerator(OutPtFeatGenerator):
    def __init__(self, half_wh, radius=None, **kwargs):
        self.half_wh = get_wh_tuple(half_wh)
        super().__init__(half_wh, **kwargs)
        self.get_center_size = self.get_center_half_wh

    def get_chosen(self, pts, centers, half_wh, gt_r_pts, stride):
        """
            pts: (num_pts=H*W, 3)
            gt_r_pts: (num_gt, num_refine, 2)
            centers: (num_gt, 1/num_refine, 2)
            wh: (num_gt, 2)
        """
        # 1. we define anchor point in circle(center=gt,radius=stride*dynamic_near_gt_r) as near gt points,
        # which can't be negative
        gts = gt_r_pts.flatten(0, 1)  # (num_gt*num_refine, 2)
        dist = torch.cdist(pts[..., :2], gts, p=2)  # (H*W, num_gt*num_refine)
        not_near_gts = dist.min(dim=1)[0] >= stride * self.near_gt_r
        # 2. outside circle(center, radius) can be negative
        half_w, half_h = half_wh[:, 0], half_wh[:, 1]
        centers = centers.flatten(0, 1)  # (num_gt * num_refine, 2)
        # dist = torch.cdist(pts[..., :2], centers, p=2)  # (H*W, num_gt*num_refine)
        chosen = ((pts[..., 0:1] - centers[..., 0]).abs() >= stride * half_w) | \
                 ((pts[..., 1:2] - centers[..., 1]).abs() >= stride * half_h)  # (H*W, num_gt*num_refine)
        chosen = torch.all(chosen, dim=1) & not_near_gts
        return chosen


OutGridCirclesPtFeatGenerator = OutCirclePtFeatGenerator
OutGridRectPtFeatGenerator = OutRectPtFeatGenerator


class GridPtFeatGenerator(AnchorPtFeatGenerator):
    def __init__(self, size, max_pos_num=-1, debug_interpolation_offset=None, *args, **kwargs):
        super(GridPtFeatGenerator, self).__init__(size, *args, **kwargs)
        self.max_pos_num = max_pos_num
        # None  => not use interpolation feat;
        # == 0 => use interpolation feat, no extra offset;
        # not None && !=0  => use interpolation feat with some offset;
        self.debug_interpolation_offset = debug_interpolation_offset

    def temp_debug_check(self, feats, pos_pts, stride, bag_feats, valid):
        if self.debug_interpolation_offset is not None:   # use interpolation feat with offset
            pos_pts[..., :2] += self.debug_interpolation_offset
        bag_feats2 = self.extract_feat(feats, pos_pts, stride)
        if self.debug_interpolation_offset is None or self.debug_interpolation_offset == 0:
            err = max([((f1 - f2).abs() * valid.float()).max() for f1, f2 in zip(bag_feats, bag_feats2)])
            assert err < 1e-3, err

        if self.debug_interpolation_offset is None:  # not use interpolation feat
            return bag_feats
        else:
            return bag_feats2

    def generate(self, feats, stride, fpn_lvl, input_data_img: Input):
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
        centers = input_data_img.gt_r_pts
        # 1. get feature map and grid pts
        # pt_feats: list[Tensor], [k, (H, W, C)]
        # pts: (H, W, 3), 3 is (x, y, lvl)
        # valid: (H, W, 1)
        pt_feats, pts, valid, stride = super().generate(feats, stride, fpn_lvl, input_data_img)
        h, w, _ = valid.shape

        # 2. get chosen flag of each gt on grid map
        # chosens: (num_gt, H, W), bool, chosens[gt_i, y, x]=True means (x, y) is chosen for gt_i
        chosens, max_pos_num = self.get_chosen_neighbours(pts, centers, stride, input_data_img)

        # 3. get feature/ pts/ valid of chosen points
        num_gts, num_refine, _ = centers.shape
        max_pos_num += num_refine
        # # print('max_pos_num', max_pos_num)

        ret_data, valid = sample_by_flag([pts] + pt_feats, chosens, max_pos_num, ret_valid=True)
        pos_pts, bag_feats = ret_data[0], ret_data[1:]  # (num_gt, max_pos_num, 3), [k, (num_gt, max_pos_num, C)]
        valid = valid.unsqueeze(dim=-1)  # (num_gt, max_pos_num, 1)
        bag_feats = self.temp_debug_check(feats, pos_pts, stride, bag_feats, valid)

        # 4. get feats of center, insert at head of return results.
        #   why flip: centers.shape=(num_gts, num_refine, 2), where centers[:, 0, :] is annotated point,
        #         flip to make it to centers[:, -1, :], that means the last one of return data is annotated point.
        centers_feats = self.extract_feat(feats, centers, stride)  # [k, (num_gts, num_refine, C)]
        for k, bag_feat in enumerate(bag_feats):
            bag_feats[k] = torch.cat([bag_feat, centers_feats[k].flip(dims=(1,))], dim=1).unsqueeze(dim=1)
        centers = self.append_other_info(centers, stride, valid)
        pos_pts = torch.cat([pos_pts, centers.flip(dims=(1,))], dim=1)

        centers_valid = torch.ones(num_gts, num_refine, 1).bool().to(valid.device)
        centers_valid = centers_valid.float().flip(dims=(1,)).bool()  # turn to float for lower version pytorch support
        valid = torch.cat([valid, centers_valid], dim=1)
        return bag_feats, pos_pts.unsqueeze(dim=1), valid.unsqueeze(dim=1)

    def extract_feat(self, feats, pts, stride):
        """
        Args:
            feats: shape=[k, (1, C, H, W)]
            pts: shape=(..., num_chosen, 2)
            stride: float
        Returns:
            sampled_feats: shape=[k, (..., num_chosen, C)]
        """
        sampled_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
        return sampled_feats

    def get_chosen_neighbours(self, pts, centers, stride, input_data_img):
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
        super().__init__(radius, **kwargs)
        self.get_center_size = self.get_center_radius

    def get_chosen_neighbours(self, pts, centers, stride, input_data_img):
        """
            get points in grid circle
            Args:
                pts:(H*W, 3)
                centers: (num_gts, num_refine, 2)
            Returns:
                chosens: (num_gts, H, W)
        """
        if self.use_dynamic_size:
            max_radius = input_data_img.other_data["max_size_in_batch"]
            centers, radius = self.get_center_size(input_data_img, stride)
            radius = radius.reshape(-1, 1, 1, 1)  # (num_gt, H, W, num_refine)
        else:
            radius = max_radius = self.radius

        H, W, _ = pts.shape
        num_gts, num_refine, _ = centers.shape

        # (1, H, W, 1, 2) - (num_gt, 1, 1, num_refine, 2) => (num_gt, H, W, num_refine, 2) => normalize =>
        # (num_gt, H, W, num_refine)
        pts_xy = pts[..., :2].reshape(1, H, W, 1, 2)
        centers = centers.reshape(num_gts, 1, 1, num_refine, 2)
        dis = torch.norm(pts_xy - centers, p=2, dim=-1)
        chosens = torch.any(dis <= radius * stride, dim=-1)  # in any one of circles => (num_gt, H, W)
        return chosens, self.get_max_pos_num(max_radius)

    def get_max_pos_num(self, radius):
        if self.max_pos_num <= 0:
            return (2 * radius + 1) ** 2
        else:
            return self.max_pos_num


class GridRectsPtFeatGenerator(GridPtFeatGenerator):
    def __init__(self, half_wh, **kwargs):
        self.half_wh = get_wh_tuple(half_wh)
        super().__init__(half_wh, **kwargs)
        self.get_center_size = self.get_center_radius

    def get_chosen_neighbours(self, pts, centers, stride, input_data_img):
        """
            get points in grid circle
            Args:
                pts:(H*W, 3)
                centers: (num_gts, num_refine, 2)
            Returns:
                chosens: (num_gts, H, W)
        """
        H, W, _ = pts.shape
        num_gts, num_refine, _ = centers.shape

        if self.use_dynamic_size:
            max_half_wh = input_data_img.other_data["max_size_in_batch"]
            centers, half_wh = self.get_center_size(input_data_img, stride)
            half_wh = half_wh.reshape(-1, 1, 1, 1, 2)  # (num_gt, H, W, num_refine, 2)
        else:
            max_half_wh = self.half_wh
            half_wh = pts.new_tensor([self.half_wh] * num_gts)

        # (1, H, W, 1, 2) - (num_gt, 1, 1, num_refine, 2) => (num_gt, H, W, num_refine, 2) => normalize =>
        # (num_gt, H, W, num_refine)
        pts_xy = pts[..., :2].reshape(1, H, W, 1, 2)
        centers = centers.reshape(num_gts, 1, 1, num_refine, 2)
        inside = ((pts_xy[..., 0] - centers[..., 0]).abs() <= half_wh[..., 0] * stride) & \
                 ((pts_xy[..., 1] - centers[..., 1]).abs() <= half_wh[..., 1] * stride)
        chosens = torch.any(inside, dim=-1)  # in any one of rects, (num_gts, H, W)
        return chosens, self.get_max_pos_num(max_half_wh)

    def get_max_pos_num(self, half_wh):
        if self.max_pos_num <= 0:
            return (half_wh[0] * 2 + 1) * (half_wh[1] * 2 + 1)
        else:
            return self.max_pos_num


class SamplePtFeatGenerator(PtFeatGenerator):
    def __init__(self, size, debug_interpolation_offset=None, **kwargs):
        super(SamplePtFeatGenerator, self).__init__(size, **kwargs)

        # None || == 0 => no extra offset;
        # not None && !=0  => use interpolation feat with some offset;
        self.debug_interpolation_offset = debug_interpolation_offset

    def get_point_neighbours(self, centers, stride, radius, max_radius):
        """
            centers: (num_gt, num_refine, 2)
            radius: (num_gt, ) / (num_gt, k)
        """
        raise NotImplementedError

    def tmp_debug_check(self, pts):
        if self.debug_interpolation_offset is not None:
            pts = pts[..., :2] + self.debug_interpolation_offset
        return pts

    def generate(self, feats, stride, fpn_lvl, input_data_img: Input):
        """
        Args:
            feats: list[Tensor], (k, (1, C, H, W))
            img_meta: dict
            gt_r_pts: (num_gts, num_refine, 2)
        Returns:
            tuple of point feats, pts(x, y, lvl, is_valid)
            bag_feats: list[Tensor], [k, (num_centers, num_refine, num_chosen, C)]
            pts: (num_centers, num_refine, num_chosen, 3), 3 is (x, y, lvl)
            valid: (num_centers, num_refine, num_chosen, num_class)
        """
        img_meta, gt_r_pts = input_data_img.img_metas, input_data_img.gt_r_pts

        centers, sizes = self.get_center_size(input_data_img, stride)
        if not self.use_dynamic_size:  # fixed sizes
            assert torch.all(
                sizes[1:] - sizes[:1] == 0), f"not use dynamic size, the sizes must be same, but got {sizes}"
            max_size = sizes[0]
        else:
            max_size = input_data_img.other_data["max_size_in_batch"]

        num_gts, num_refine, _ = centers.shape
        pts, valid = self.get_point_neighbours(centers, stride, sizes, max_size)
        assert torch.all(gt_r_pts[:, 0] == input_data_img.gt_pts)

        pts, valid = self.append_gt(pts, valid, gt_r_pts[:, 0])
        valid = self.get_point_valid(pts, *img_meta['pad_shape'][:2])[..., None] & valid
        pts = self.append_other_info(pts, stride, valid)
        # bag_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
        bag_feats = [self.extract_point_feat(feat, self.tmp_debug_check(pts[..., :2]), stride) for feat in feats]
        return bag_feats, pts, valid

    def append_gt(self, pts, valid, gts):
        """
            pts: shape=(num_gt, num_refine, num_chosen, 2)
            gts: shape=(num_gt, 2)
        Return:
            pts[:, :, -1, :] is gt, and pts[:, :, -2, :] is center
            while gt == center, set valid[:, :, -2] = False
        """
        gts = gts.unsqueeze(dim=1).unsqueeze(dim=1)
        pts = torch.cat([pts, gts], dim=2)
        centers = pts[:, :, -1:]
        center_valid = torch.any((centers != gts), dim=-1).unsqueeze(dim=-1)  # (num_gt, num_refine, 1, 1)
        chosen_pt_valid, gt_valid = valid[:, :, :-1], valid[:, :, -1:]
        valid = torch.cat([chosen_pt_valid, center_valid, gt_valid], dim=2)
        return pts, valid


class CirclePtFeatGenerator(SamplePtFeatGenerator):
    def __init__(self, radius, start_angle=0, base_num_point=8, same_num_all_radius=False, **kwargs):
        self.radius = radius
        self.start_angle = start_angle
        self.base_num_point = base_num_point
        self.same_num_all_radius = same_num_all_radius

        super().__init__(radius, **kwargs)
        self.depends = dict()
        self.get_center_size = self.get_center_radius

    # def generate(self, feats, stride, fpn_lvl, input_data_img: Input):
    #     """
    #     Args:
    #         feats: list[Tensor], (k, (1, C, H, W))
    #         img_meta: dict
    #         centers: (num_gts, num_refine, 2)
    #     Returns:
    #         tuple of point feats, pts(x, y, lvl, is_valid)
    #         bag_feats: list[Tensor], [k, (num_centers, num_refine, num_chosen, C)]
    #         pts: (num_centers, num_refine, num_chosen, 3), 3 is (x, y, lvl)
    #         valid: (num_centers, num_refine, num_chosen, num_class)
    #     """
    #     img_meta, centers = input_data_img.img_metas, input_data_img.gt_r_pts
    #     num_gts, num_refine, _ = centers.shape
    #     pts, _ = self.get_point_neighbours(centers, stride, self.radius)
    #
    #     valid = self.get_point_valid(pts, *img_meta['pad_shape'][:2])
    #     pts = self.append_other_info(pts, stride, valid)
    #     bag_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
    #     # valid_class = pts.new_zeros(*valid.shape, self.num_classes)  # repeat valid to num_class
    #     # valid_class[..., :] = valid[..., None]
    #     return bag_feats, pts, valid[..., None]

    def get_point_neighbours(self, centers, stride, radius, max_radius):
        """
        Args:
            centers: Tensor, shape=(num_gt, num_refine, 2), each point is record as (x, y)
            stride: int
            radius: shape=(num_gt, )
            max_radius: int
        Returns:
            choose points inside circle which radius is r * stride with gt_point as center.
            chosen_pts: shape=(num_gt, num_refine, num_chosen, 2)
            valid: shape=(num_gt, num_refine, num_chosen, 1)
        """
        assert len(centers.shape) == 3
        num_gts, num_refine, _ = centers.shape
        centers = centers.flatten(0, 1)

        # 1. get points of each circle round
        assert torch.all(radius >= 1), f"{radius} give for radius."
        chosen_pts, radius2count = [], [0] * (max_radius + 1)
        for i in range(max_radius):
            r = (i + 1) * stride
            num_pts = self.base_num_point if self.same_num_all_radius else (self.base_num_point * (i + 1))
            angles = torch.arange(num_pts).float().to(centers.device) / num_pts * 360 + self.start_angle
            angles = angles / 360 * np.pi * 2
            anchor_pts = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
            chosen_pts.append(anchor_pts)
            radius2count[i + 1] = radius2count[i] + num_pts
        chosen_pts = torch.cat(chosen_pts).unsqueeze(dim=0) + centers.reshape(-1, 1, 2)

        chosen_pts = torch.cat([chosen_pts, centers.unsqueeze(dim=1)], dim=1)  # append center at end

        chosen_pts = chosen_pts.reshape(num_gts, num_refine, -1, 2)
        radius2count = torch.LongTensor(radius2count).to(chosen_pts.device)

        # 2. set pt out of circle as invalid
        valid = torch.ones(*chosen_pts.shape[:3], 1).bool().to(chosen_pts.device)
        for gt_i, r in enumerate(radius):
            valid[gt_i, :, radius2count[r]:] = False
            valid[gt_i, :, -1] = True  # last one is appended center
        return chosen_pts, valid


# class DynamicCirclePtFeatGenerator(CirclePtFeatGenerator):
#     def __init__(self, radius=-1.0, **kwargs):
#         super(DynamicCirclePtFeatGenerator, self).__init__(radius, **kwargs)
#         self.get_center_size = self.get_center_radius
#
#     def generate(self, feats, stride, fpn_lvl, input_data_img: Input):
#         """
#         Args:
#             feats: list[Tensor], (k, (1, C, H, W))
#             img_meta: dict
#             gt_r_pts: (num_gts, num_refine, 2)
#         Returns:
#             tuple of point feats, pts(x, y, lvl, is_valid)
#             bag_feats: list[Tensor], [k, (num_centers, num_refine, num_chosen, C)]
#             pts: (num_centers, num_refine, num_chosen, 3), 3 is (x, y, lvl)
#             valid: (num_centers, num_refine, num_chosen, num_class)
#         """
#         img_meta, gt_r_pts = input_data_img.img_metas, input_data_img.gt_r_pts
#         max_radius = input_data_img.other_data["max_size_in_batch"]
#
#         if not self.use_dynamic_size:   # fixed radius
#             return super().generate(feats, stride, fpn_lvl, input_data_img)
#
#         centers, radius = self.get_center_size(input_data_img, stride)
#         num_gts, num_refine, _ = centers.shape
#         pts, valid = self.get_each_point_neighbours(centers, stride, radius, max_radius)
#         assert torch.all(gt_r_pts[:, 0] == input_data_img.gt_pts)
#
#         pts, valid = self.append_gt(pts, valid, gt_r_pts[:, 0])
#         valid = self.get_point_valid(pts, *img_meta['pad_shape'][:2])[..., None] & valid
#         pts = self.append_other_info(pts, stride, valid)
#         bag_feats = [self.extract_point_feat(feat, pts[..., :2], stride) for feat in feats]
#         return bag_feats, pts, valid
#
#     def append_gt(self, pts, valid, gts):
#         """
#             pts: shape=(num_gt, num_refine, num_chosen, 2)
#             gts: shape=(num_gt, 2)
#         Return:
#             pts[:, :, -1, :] is gt, and pts[:, :, -2, :] is center
#             while gt == center, set valid[:, :, -2] = False
#         """
#         gts = gts.unsqueeze(dim=1).unsqueeze(dim=1)
#         pts = torch.cat([pts, gts], dim=2)
#         centers = pts[:, :, -1:]
#         center_valid = torch.any((centers != gts), dim=-1).unsqueeze(dim=-1)  # (num_gt, num_refine, 1, 1)
#         chosen_pt_valid, gt_valid = valid[:, :, :-1], valid[:, :, -1:]
#         valid = torch.cat([chosen_pt_valid, center_valid, gt_valid], dim=2)
#         return pts, valid
#
#     def get_each_point_neighbours(self, centers, stride, radius, max_radius):
#         """
#         Args:
#             centers: Tensor, shape=(num_gt, num_refine, 2), each point is record as (x, y)
#             stride:
#             radius: shape(num_gts, )
#         Returns:
#             choose points inside circle which radius is r * stride with gt_point as center.
#             chosen_pts: shape=(num_gt, num_refine, num_chosen, 2)
#             valid: shape=(num_gt, num_refine, num_chosen, 1)
#         """
#         assert len(centers.shape) == 3
#         assert len(radius.shape) == 1
#         # max_radius = radius.max()
#
#         # 1. get points of each circle round
#         chosen_pts, radius2count = self.get_point_neighbours(centers, stride, max_radius)
#
#         # 2. set pt out of circle as invalid
#         valid = torch.ones(*chosen_pts.shape[:3], 1).bool().to(chosen_pts.device)
#         for gt_i, r in enumerate(radius):
#             valid[gt_i, :, radius2count[r]:] = False
#             valid[gt_i, :, -1] = True   # last one is appended center
#         return chosen_pts, valid


class RectPtFeatGenerator(SamplePtFeatGenerator):
    def __init__(self, half_wh, radius=None, **kwargs):
        self.half_wh = get_wh_tuple(half_wh)
        super().__init__(half_wh, **kwargs)
        self.get_center_size = self.get_center_half_wh

    def get_point_neighbours(self, centers, stride, half_wh, max_half_wh):
        """
        Args:
            centers: Tensor, shape=(num_gt, num_refine, 2), each point is record as (x, y)
            stride: int
            half_wh: (num_gt, 2)
            max_half_wh: (int, int)
        Returns:
            choose points inside circle which radius is r * stride with gt_point as center.
            chosen_anchor_pts: shape=(num_gt_pts, num_refine, num_chosen, 2)
            valid: shape=(num_gt, num_refine, num_chosen, 1)
        """
        assert len(centers.shape) == 3
        num_gts, num_refine, _ = centers.shape
        centers = centers.flatten(0, 1)  # (num_gt*num_refine, 2)

        assert torch.all(half_wh >= 1), f"{half_wh} give for wh."
        max_half_w, max_half_h = max_half_wh

        # 1. generate sample points
        pt_x = torch.arange(-max_half_w, max_half_w + 1, device=centers.device).type_as(centers)
        pt_y = torch.arange(-max_half_h, max_half_h + 1, device=centers.device).type_as(centers)
        pt_y, pt_x = torch.meshgrid(pt_y, pt_x)
        pts_xy = torch.stack((pt_x, pt_y), dim=-1)
        valid = torch.zeros(num_gts, num_refine, *pts_xy.shape[:-1], device=pts_xy.device, dtype=torch.bool)
        pts_xy = pts_xy.reshape(-1, 2).unsqueeze(dim=0) * stride  # (1, num_chosen, 2)
        chosen_pts = pts_xy + centers.reshape(-1, 1, 2)  # (num_gt*num_refine, num_chosen, 2)

        # 2. set valid
        for gt_i, (half_w_i, half_h_i) in enumerate(half_wh):
            valid[gt_i, :, max_half_h - half_h_i:max_half_h + half_h_i,
                  max_half_w - half_w_i:max_half_w + half_w_i] = True
        valid = valid.reshape(num_gts * num_refine, -1)  # (num_gt*num_refine, num_chosen)

        # 3. add center
        chosen_pts = torch.cat([chosen_pts, centers.unsqueeze(dim=1)], dim=1)  # append center at end
        centers_valid = torch.ones(centers.shape[0], 1, dtype=torch.bool, device=valid.device)
        valid = torch.cat([valid, centers_valid], dim=-1)  # (num_gt*num_refine, num_chosen)

        chosen_pts = chosen_pts.reshape(num_gts, num_refine, -1, 2)
        valid = valid.reshape(num_gts, num_refine, -1, 1)
        return chosen_pts, valid


import matplotlib.pyplot as plt


class TestSampler(object):
    def __init__(self, debug_cfg):
        self.debug_cfg = debug_cfg

    def show_pos(self, featmaps, feats, pts, valid, input_data_img: Input):
        """
        feats: shape=[k, (num_gt, num_refine, num_chosen, C)]
            pts: shape=(num_gt, num_refine, num_chosen, 3)
            valid: shape=(num_gt, num_refine, num_chosen, 1)
        """
        img_meta = input_data_img.img_metas

        pts = pts[valid.squeeze(dim=-1)].cpu().numpy()
        pts += 1  # add 1 pixel offset for better view of overlap point with neg.
        x, y = pts[..., 0].reshape(-1), pts[..., 1].reshape(-1)
        self.plot_img(img_meta)
        plt.scatter(x, y, s=5, c='red')

    def show_neg(self, featmaps, feats, pts, valid, input_data_img: Input):
        """
            feats: shape=[k, (H, W, C)] or [k, (H*W, C)]
            pts: shape=(H, W, 3) or (H*W, 3)
            valid: shape=(H, W, num_class) or (H*W, num_class)
        """
        img_meta, gt_labels = input_data_img.img_metas, input_data_img.gt_labels

        pts = pts[valid[..., gt_labels[0]]].cpu().numpy()
        x, y = pts[..., 0].reshape(-1), pts[..., 1].reshape(-1)
        plt.scatter(x, y, s=5, c='black')

    def show(self, featmaps, feats, pts, valid, input_data_img: Input):
        if not self.debug_cfg.get('open', False) or not self.debug_cfg.get("show_sampler", False):
            return
        true_bbox = input_data_img.gt_true_bboxes
        size = ((true_bbox[:, 2] - true_bbox[:, 0]) * (true_bbox[:, 3] - true_bbox[:, 1])) ** 0.5
        if (size < 96).all():
            return
        if len(pts.shape) == 4:
            self.show_pos(featmaps, feats, pts, valid, input_data_img)
        elif len(pts.shape) == 3 or len(pts.shape) == 2:
            self.show_neg(featmaps, feats, pts, valid, input_data_img)
            plt.show()
        else:
            raise ValueError

    def plot_img(self, img_meta):
        img_path = img_meta['filename']
        sw, sh = img_meta['scale_factor'][:2]
        from PIL import Image
        from ssdcv.plot_paper.plt_paper_config import set_plt
        import matplotlib.pyplot as plt

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
