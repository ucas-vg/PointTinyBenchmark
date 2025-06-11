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


class PointFeatureExtractor(BaseModule):
    def __init__(self, strides: tuple, neighbour_cfg, neg_cfg=dict(scale=3.), align_corners=True, init_cfg=None):
        super(PointFeatureExtractor, self).__init__(init_cfg)
        self.neighbour_cfg = neighbour_cfg
        self.align_corners = align_corners
        self.strides = strides
        # PointGenerator have cache inside, can share for different FPN level
        self.point_generators = [PointGenerator() for _ in strides]

        nearest_k = self.neighbour_cfg.get("nearest_k", -1)
        neighbour_radius = self.neighbour_cfg.get("neighbour_radius", -1)
        assert sum([nearest_k > 0, neighbour_radius > 0]) == 1
        self.neg_cfg = neg_cfg

    def forward(self, all_feats, gt_points_list, img_metas_list, gt_points_ignore_list=None, fpn_gt_idx=None):
        """
        Args:
            all_feats:              list[Tensor], shape=[num_feat, num_level, (B, C, H, W)].
                                    num_feat==2 means cls_feats and ins_feats in all_feats
            gt_points_list:         list[Tensor], shape=[B, (num_gt, 2)]
            img_metas_list:         list[dict], shape=[B]
            gt_points_ignore_list:  list[Tensor], shape=[B, (num_gt_ignore, 2)]
            fpn_gt_idx:             list[list[Tensor]], shape=[B, num_level, (num_gt_level, 2)]
        Returns:
            point_bag_feats:
            point_gt_feats:
        """
        featmap_sizes = [f.shape[2:] for f in all_feats[0]]
        device = all_feats[0][0].device
        anchor_pts_list, valid_flags_list = self.get_anchor_points(featmap_sizes, img_metas_list, device)

        if fpn_gt_idx is None:
            # if fpn_gt_idx not given, extract features on all FPN level for each gt
            # [B, num_level, (num_gt, 2)]
            fpn_gt_idx = [[torch.arange(len(gt_points_img)).long() for _ in self.strides]
                          for gt_points_img in gt_points_list]

        num_feat = len(all_feats)
        chosen_pts_all, bag_all_feats = [], [[] for _ in all_feats]
        for img_id, gt_points_img in enumerate(gt_points_list):
            chosen_pts_img = [[] for _ in gt_points_img]
            bag_all_feats_img = [[[] for _ in gt_points_img] for _ in all_feats]
            img_meta = img_metas_list[img_id]
            for lvl, stride in enumerate(self.strides):
                gt_idx = fpn_gt_idx[img_id][lvl]
                if len(gt_idx) == 0:
                    continue
                gt_points = gt_points_list[img_id][gt_idx]
                # img_metas = [img_metas_list[img_id][gt_i] for gt_i in gt_idx]
                # gt_points_ignore = gt_points_ignore_list[img_id][gt_idx]
                anchor_pts, valid_flags = anchor_pts_list[img_id][lvl], valid_flags_list[img_id][lvl]
                anchor_pts = anchor_pts[valid_flags]

                # (num_gts_lvl, num_chosen, 2)
                chosen_pts = self.get_point_neighbours(anchor_pts, gt_points)
                is_valid = self.get_point_valid(chosen_pts, *img_meta['ori_shape'][:2]).unsqueeze(dim=-1).type_as(chosen_pts)
                # add fpn level
                chosen_pts_lvl = torch.full(chosen_pts.shape[:-1], lvl, dtype=chosen_pts.dtype).to(chosen_pts.device)
                chosen_pts = torch.cat([chosen_pts, chosen_pts_lvl.unsqueeze(-1), is_valid], dim=-1)

                the_bag_all_feats = []
                for k in range(num_feat):
                    feat = all_feats[k][lvl][img_id:img_id + 1]
                    # (num_gts_lvl, num_chosen, feat_channel)
                    the_bag_feats = self.extract_point_feat(feat, chosen_pts[..., :2], gt_points, self.strides[lvl])
                    the_bag_all_feats.append(the_bag_feats)

                # map back to origin gt, point_gt_feats_lvl: [num_gt, use_level, (num_chosen, feat_channel)]
                for i, gt_i in enumerate(gt_idx):
                    chosen_pts_img[gt_i].append(chosen_pts[i])
                    for k in range(num_feat):
                        bag_all_feats_img[k][gt_i].append(the_bag_all_feats[k][i])
            # [num_gt, use_level, (num_chosen, feat_channel)] => (num_gt, num_chosen_all_lvl, feat_channel)
            for gt_i in range(len(chosen_pts_img)):
                chosen_pts_img[gt_i] = torch.cat(chosen_pts_img[gt_i])
                for k in range(num_feat):
                    bag_all_feats_img[k][gt_i] = torch.cat(bag_all_feats_img[k][gt_i])
            chosen_pts_img = torch.stack(chosen_pts_img)
            bag_all_feats_img = [torch.stack(bag_all_feats_img[k]) for k in range(num_feat)]

            chosen_pts_all.append(chosen_pts_img)
            for k in range(num_feat):
                bag_all_feats[k].append(bag_all_feats_img[k])  # [B, (num_gt, num_chosen_all_lvl, feat_channel)]

        chosen_pts_all = torch.cat(chosen_pts_all, dim=0)
        for k in range(num_feat):
            # => (num_gts_all_img, num_chosen_all_level, feat_channel)
            bag_all_feats[k] = torch.cat(bag_all_feats[k], dim=0)
        return chosen_pts_all, bag_all_feats

    def get_anchor_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
        Returns:
            tuple: points of each image, valid flags of each image
            points_list: shape=[B, num_level, (num_pts, 3)]
            valid_flag_list: shape=[B, num_level, (num_pts, )]
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.strides[i], device)
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['ori_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def get_point_neighbours(self, anchor_pts_stride: torch.Tensor, gt_pts):
        """
        Args:
            anchor_pts_stride: Tensor, shape=(num_anchor_pts, 3), anchor point is (x, y, stride)
            gt_pts: Tensor, shape=(num_gt_pts, 2), each point is record as (x, y)
        Returns:
            choose points inside circle which radius is r * stride with gt_point as center.
            choose closest k point as neighbours

            chosen_anchor_pts: shape=(num_gt_pts, num_chosen, 2)
        """
        nearest_k = self.neighbour_cfg.get("nearest_k", -1)
        neighbour_radius = self.neighbour_cfg.get("neighbour_radius", -1)

        if nearest_k > 0:
            anchor_pts = anchor_pts_stride[:, :2].reshape(1, -1, 2)
            distance = (gt_pts.reshape(-1, 1, 2) - anchor_pts).norm(dim=-1)  # (num_gt, num_anchor)
            assert nearest_k <= 2 * distance.shape[1], f"{nearest_k} vs {distance.shape[1]}"
            k = min(nearest_k, distance.shape[1])
            _, indices = torch.topk(distance, k, dim=1, largest=False)
            if k < nearest_k:
                indices = torch.cat([indices, indices[:, k-nearest_k:]], dim=-1)   # add farthest point
            # (num_gt, nearest_k, 2)
            chosen_pts = torch.stack([anchor_pts_stride[idx, :2] for i, idx in enumerate(indices)])

        elif neighbour_radius > 0:
            anchor_lvl = anchor_pts_stride[:, 2].reshape(-1)
            assert (anchor_lvl == anchor_lvl[0]).all(), "must be same fpn level"
            stride = anchor_lvl[0]
            start_angle = self.neighbour_cfg.get("start_angle", 0)
            base_num_point = self.neighbour_cfg.get("base_num_point", 8)
            same_num_all_radius = self.neighbour_cfg.get("same_num_all_radius", False)

            chosen_pts = []
            for i in range(neighbour_radius):
                r = (i + 1) * stride
                num_pts = base_num_point if same_num_all_radius else (base_num_point * (i+1))

                angles = torch.arange(num_pts).float().to(gt_pts.device) / num_pts * 360 + start_angle
                angles = angles / np.pi * 2
                anchor_pts = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
                chosen_pts.append(anchor_pts)
            chosen_pts = torch.cat(chosen_pts).unsqueeze(dim=0) + gt_pts.reshape(-1, 1, 2)
        else:
            raise ValueError

        # add gt to last one
        chosen_pts = torch.cat([chosen_pts, gt_pts.unsqueeze(dim=1)], dim=1)
        return chosen_pts

    def get_point_valid(self, pts, valid_h, valid_w):
        """
        Args:
            pts: shape=(..., 2)
        Returns:
        """
        valid = torch.zeros(pts.shape[:-1], dtype=torch.bool, device=pts.device)
        valid[(0 <= pts[..., 0]) & (pts[..., 0] < valid_w) & (0 <= pts[..., 1]) & (pts[..., 1] < valid_h)] = 1
        return valid

    def extract_point_feat(self, feat, chosen_pts, gt_pts, stride):
        """
        Args:
            feat: shape=(1, C, H, W)
            chosen_pts: shape=(num_gt_pts, num_chosen, 2)
            gt_pts: shape=(num_gt_pts, 2)
            stride: float
        Returns:
            point_bag_feats: shape=(num_gts, num_chosen, feat_channel)
            point_gt_feats: shape=(num_gts, 1, feat_channel)
        """
        chosen_pts = chosen_pts.unsqueeze(0) / stride  # => (B=1, num_gt_pts, num_chosen, 2)
        # permute(0, 2, 3, 1)[0]: (B=1, feat_c, num_gt_pts, num_chosen) => (num_gt_pts, num_chosen, feat_c)
        bag_feats = self.grid_sample(feat, chosen_pts).permute(0, 2, 3, 1)[0]
        if self.neighbour_cfg.get("nearest_k", -1) > 0:
            TestCPRHead.test_extract_point_feat(chosen_pts[:, :-1], feat, bag_feats[:, :-1])

        # gt_pts = gt_pts.view(1, -1, 1, 2) / stride
        # gt_feats = self.grid_sample(feat, gt_pts).permute(0, 2, 3, 1)[0]
        return bag_feats

    def grid_sample(self, feat, chosen_pts):
        """
        # (B=1, num_gt_pts, num_chosen, 2)
        Args:
            feat: shape=(B, C, H, W)
            chosen_pts:  shape=(B, num_gts, num_chosen, 2)
        Returns:
        """
        if self.align_corners:
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
        return F.grid_sample(feat, chosen_pts, align_corners=self.align_corners, padding_mode=padding_mode)

    def num_sample_per_lvl(self):
        nearest_k = self.neighbour_cfg.get("nearest_k", -1)
        neighbour_radius = self.neighbour_cfg.get("neighbour_radius", -1)

        if nearest_k > 0:
            return nearest_k + 1  # + 1 is means gt
        if neighbour_radius > 0:
            same_num_all_radius = self.neighbour_cfg.get("same_num_all_radius", False)
            base_num_point = self.neighbour_cfg.get("base_num_point", 8)
            if same_num_all_radius:
                return neighbour_radius * base_num_point + 1
            else:
                return np.sum(np.arange(neighbour_radius+1)) * base_num_point + 1
        assert False

    def get_neg_points_feat(self, all_feats, img_metas_list, gt_pts_list):
        def anchor_points(h, w, valid_h, valid_w, stride, device):
            x, y = torch.arange(w).to(device), torch.arange(h).to(device)
            y, x = torch.meshgrid(y, x)
            y, x = y.reshape(-1), x.reshape(-1)
            pts = torch.stack([x, y], dim=-1) * stride + stride / 2
            return pts, y, x, self.get_point_valid(pts, valid_h, valid_w)

        def stack_feats(feats):
            """
            Args:
                feats: [B, (n, c)]
            Returns:
            """
            max_num = max([len(f) for f in feats])
            s0 = feats[0].shape
            for f in feats:
                assert f.shape[1:] == s0[1:], f"{f.shape} vs {s0}"

            shape = (max_num, ) + feats[0].shape[1:]
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

        device = all_feats[0][0].device
        featmap_sizes = [f.shape[2:] for f in all_feats[0]]
        all_neg_feats = [[[] for _ in range(len(img_metas_list))] for _ in range(len(all_feats))]
        for im_id, img_meta in enumerate(img_metas_list):
            for lvl, stride in enumerate(self.strides):
                pts, y, x, valid = anchor_points(*featmap_sizes[lvl], *img_meta['ori_shape'][:2], stride, device)

                dist = torch.cdist(pts, gt_pts_list[im_id], 2)  # (n, 2) vs (m, 2)
                chosen_neg = dist.min(dim=1)[0] >= stride * self.neg_cfg['scale']

                valid = valid & chosen_neg
                y, x, valid = y[chosen_neg], x[chosen_neg], valid[chosen_neg]
                y, x = y[valid], x[valid]
                for i, feats in enumerate(all_feats):
                    all_neg_feats[i][im_id].append(feats[lvl][im_id, :, y, x].permute(1, 0))  # (num_neg, feat_c)
            for i, _ in enumerate(all_feats):
                all_neg_feats[i][im_id] = torch.cat(all_neg_feats[i][im_id])
        for i, _ in enumerate(all_feats):
            all_neg_feats[i], valids = stack_feats(all_neg_feats[i])
        return all_neg_feats, valids


def swap_list_order(alist):
    """
    Args:
        alist: shape=(B, num_level, ....)
    Returns:
        alist: shape=(num_level, B, ...)
    """
    new_order0 = len(alist[0])
    return [[alist[i][j] for i in range(len(alist))] for j in range(new_order0)]


@HEADS.register_module()
class CPRHead2(AnchorFreeHead):
    """
    Coarse Point Refine Head
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_cls_fcs=0,
                 fc_out_channels=1024,
                 feature_extractor=dict(
                     align_corners=True,
                     neighbour_cfg=dict(
                         neighbour_radius=-1,
                         nearest_k=21
                     ),
                     neg_cfg=dict(
                         scale=3.,
                     ),
                 ),
                 ins_share_head_feat=True,
                 ins_share_head_classifier=False,
                 loss_mil=dict(
                     type='MILLoss',
                     use_binary=True,
                     loss_weight=1.0),
                 loss_type=0,
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
#                      override=dict(
#                         type='Normal',
#                         name='cls_out',
#                         std=0.01,
#                         bias=1)
                 ),
                 debug=False,
                 **kwargs):

        self.use_binary_cls = loss_mil.get('use_binary', False)
        self.num_cls_out = num_classes if self.use_binary_cls else num_classes + 1
        # if loss_mil['type'] == 'MIL2Loss':
        #     self.num_cls_out *= 2
        self.num_cls_fcs = num_cls_fcs
        self.fc_out_channels = fc_out_channels
        self.ins_share_head_feat = ins_share_head_feat
        self.ins_share_head_classifier = ins_share_head_classifier

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_mil,
            init_cfg=init_cfg,
            **kwargs)

        if feature_extractor is not None:
            if 'strides' not in feature_extractor:
                feature_extractor['strides'] = self.strides
            self.feature_extractor = PointFeatureExtractor(**feature_extractor)

        self.loss_mil = build_loss(loss_mil)
        self.loss_type = loss_type

        self.debug = debug
        TestCPRHead.DO_TEST = debug

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
            self.ins_out = nn.Linear(chn, self.num_cls_out)
        else:
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

    def get_cls_outs(self, cls_feat, ins_feat, gt_points, img_metas, gt_points_ignore=None):
        if self.ins_share_head_feat:
            chosen_pts_all, (bag_cls_feats, ) = self.feature_extractor(
                [cls_feat, ], gt_points, img_metas, gt_points_ignore)

            bag_cls_feats = bag_cls_feats.view(-1, bag_cls_feats.shape[-1])
            for i, cls_fc in enumerate(self.cls_fcs):
                bag_cls_feats = self.relu(self.cls_fcs[i](bag_cls_feats))
            bag_ins_feats = bag_cls_feats
        else:
            chosen_pts_all, (bag_cls_feats, bag_ins_feats) = self.feature_extractor(
                [cls_feat, ins_feat], gt_points, img_metas, gt_points_ignore)

            bag_cls_feats = bag_cls_feats.view(-1, bag_cls_feats.shape[-1])
            bag_ins_feats = bag_ins_feats.view(-1, bag_cls_feats.shape[-1])
            for i, cls_fc in enumerate(self.cls_fcs):
                bag_cls_feats = self.relu(cls_fc(bag_cls_feats))
            for i, ins_fc in enumerate(self.ins_fcs):
                bag_ins_feats = self.relu(ins_fc(bag_ins_feats))

        num_gts_all_img = sum([len(gts) for gts in gt_points])
        bag_cls_outs = self.cls_out(bag_cls_feats).view(num_gts_all_img, -1, self.num_cls_out)
        bag_ins_outs = self.ins_out(bag_ins_feats).view(num_gts_all_img, -1, self.num_cls_out)\
            if not (self.ins_share_head_feat and self.ins_share_head_classifier) else bag_cls_outs
        bag_gt_outs = torch.stack([bag_cls_outs, bag_ins_outs], dim=0)

        if self.feature_extractor.neg_cfg['scale'] > 0:
            (neg_feats,), neg_valid = self.feature_extractor.get_neg_points_feat([cls_feat, ], img_metas, gt_points)
            neg_feats = neg_feats.view(-1, neg_feats.shape[-1])
            neg_valid = neg_valid.view(-1,)
            for i, cls_fc in enumerate(self.cls_fcs):
                neg_feats = self.relu(self.cls_fcs[i](neg_feats))
            neg_outs = self.cls_out(neg_feats)
        else:
            neg_outs, neg_valid = None, None

        return chosen_pts_all, bag_gt_outs, neg_outs, neg_valid

    def loss(self, cls_feat, ins_feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, gt_true_bboxes=None):
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None
        gt_labels_all = torch.cat(gt_labels, dim=0)
        chosen_pts_all, bag_gt_outs, neg_outs, neg_valid = self.get_cls_outs(
            cls_feat, ins_feat, gt_points, img_metas, gt_points_ignore)

        if self.debug:
            points, scores, points_lvl = self.get_refine_point(bag_gt_outs, chosen_pts_all, gt_points,
                                                       gt_labels, img_metas, gt_true_bboxes)
        return getattr(self, f'loss{self.loss_type}')(bag_gt_outs, chosen_pts_all, neg_outs, neg_valid,
                                                      gt_labels_all, gt_points, gt_true_bboxes)

    def chose_fpn_lvl(self, bag_gt_outs, all_gt_labels):
        (bag_cls_outs, gt_cls_outs), (bag_ins_outs, gt_ins_outs) = self.split_bag_gt(bag_gt_outs[0], bag_gt_outs[1])

        all_gt_idx = torch.arange(len(all_gt_labels))
        num_all_gts, num_level, num_cls_out = gt_cls_outs.shape

        # step 1: choose fpn level
        prob_gt_cls = gt_cls_outs.softmax(dim=-1)
        prob_gt_lvl = gt_ins_outs.softmax(dim=-2)
        prob_gt = prob_gt_cls * prob_gt_lvl  # (num_all_gts, num_level, cls_out)
        prob_gt = prob_gt[all_gt_idx, :, all_gt_labels]

        prob_bag_cls = bag_cls_outs.softmax(dim=-1)
        prob_bag_ins = bag_ins_outs.softmax(dim=-2)
        prob_bag = prob_bag_cls * prob_bag_ins
        prob_bag = prob_bag[all_gt_idx, :, all_gt_labels]

        # # (num_all_gts, num_level)
        prob_bag, max_prob_idx_each_lvl = prob_bag.reshape(num_all_gts, num_level, -1).max(dim=-1)
        # (num_all_gts, num_level)
        prob_lvl = prob_gt * prob_bag
        lvl_max_prob, lvl_max_prob_idx = prob_lvl.max(dim=1)

        bag_gt_outs = bag_gt_outs.reshape(len(bag_gt_outs), num_all_gts, num_level, -1, num_cls_out)
        bag_gt_outs_lvl = bag_gt_outs[:, all_gt_idx, lvl_max_prob_idx, :, :]

        TestCPRHead.test_chosen_fpn_level(prob_bag_cls, prob_bag_ins, lvl_max_prob_idx,
                                          all_gt_idx, all_gt_labels, num_all_gts, num_level)

        return lvl_max_prob_idx, bag_gt_outs_lvl

    def loss0(self, bag_gt_outs, chosen_pts_all, neg_outs, neg_valid, gt_labels_all, gt_points, gt_true_bboxes=None):
        """
            Args:
                bag_cls_outs: shape=(num_gts, num_bag_all_level, cls_out)
                point_gt_outs: shape=(num_gts, num_level, cls_out)
                gt_labels_all: shape=(num_gts,)
                gt_true_bboxes: shape=(B, num_gt_per_img, 4)
            Returns:
        """
        is_valid = chosen_pts_all[..., -1:]
        loss_bag, log = self.loss_mil(bag_gt_outs[0], bag_gt_outs[1], gt_labels_all, is_valid, neg_outs, neg_valid)
        losses = {
            "loss_bag": loss_bag,
        }
        losses.update(log)

        if gt_true_bboxes is not None:
            losses.update(self.fpn_acc(bag_gt_outs, chosen_pts_all, gt_labels_all, gt_true_bboxes))

        return losses

    def loss1(self, bag_gt_outs, chosen_pts_all, neg_outs, neg_valid, gt_labels_all, gt_points, gt_true_bboxes=None):
        """
            Args:
                point_bag_outs: shape=(num_gts, num_bag_all_level, cls_out)
                point_gt_outs: shape=(num_gts, num_level, cls_out)
                gt_labels: shape=(num_gts,)
                gt_true_bboxes: shape=(B, num_gt_per_img, 4)
            Returns:
        """
        # point_bag_gt_outs = torch.cat([point_bag_outs, point_gt_outs], dim=1)
        loss_bag, bag_acc = self.loss_mil(bag_gt_outs[0], bag_gt_outs[1], gt_labels_all)
        bag_acc = bag_acc['bag_acc']

        _, bag_gt_outs_lvl = self.chose_fpn_lvl(bag_gt_outs, gt_labels_all)
        lvl_loss_bag, lvl_bag_acc = self.loss_mil(bag_gt_outs_lvl[0], bag_gt_outs_lvl[1], gt_labels_all)
        lvl_bag_acc = lvl_bag_acc['bag_acc']

        losses = {
            "bag_acc": bag_acc,
            "loss_bag": loss_bag,
            "lvl_bag_acc": lvl_bag_acc,
            "lvl_loss_bag": lvl_loss_bag,
        }
        if gt_true_bboxes is not None:
            losses.update(self.fpn_acc(bag_gt_outs, chosen_pts_all, gt_labels_all, gt_true_bboxes))
        return losses

    def fpn_acc(self, bag_gt_outs, chosen_pts_all, all_gt_labels, gt_true_bboxes):
        if not hasattr(self, 'fpn_mean'):
            self.fpn_mean = torch.full((len(self.strides),), 0., dtype=bag_gt_outs.dtype).to(bag_gt_outs.device)
            self.last_var = torch.full((len(self.strides),), 0., dtype=bag_gt_outs.dtype).to(bag_gt_outs.device)

        gt_true_bboxes = torch.cat(gt_true_bboxes)
        gt_log2_size = (gt_true_bboxes[:, 2:] - gt_true_bboxes[:, :2]).log2().mean(dim=-1)
        _, num_all_gt, _, num_cls_out = bag_gt_outs.shape
        max_fpn_lvl, _ = self.chose_fpn_lvl(bag_gt_outs, all_gt_labels)

        # vars = [None] * len(self.strides)
        # for lvl in range(len(self.strides)):
        #     lvl_size = gt_log2_size[max_fpn_lvl == lvl]
        #     if lvl_size.shape[0] > 0:
        #         self.fpn_mean[lvl] = 0.9 * self.fpn_mean[lvl] + 0.1 * lvl_size.mean()
        #         vars[lvl] = ((lvl_size - self.fpn_mean[lvl]) ** 2).mean() ** 0.5
        #         self.last_var[lvl] = vars[lvl]
        #     else:
        #         vars[lvl] = self.last_var[lvl]
        # return {
        #     f"fpn_acc{i}": std for i, std in enumerate(vars)
        # }
        rates = gt_log2_size.new_tensor([0.] * len(self.strides))
        for lvl in range(len(self.strides)):
            lvl_size = gt_log2_size[max_fpn_lvl == lvl]
            rates[lvl] = lvl_size.shape[0]
        return {
            f"fpn{i}": rate for i, rate in enumerate(rates)
        }

    def split_bag_gt(self, *point_bag_gts):
        """
        Args:
            point_bag_gts: shape=[len_list, (num_gts, num_bag_all_level, ...)]
        Returns:
        """
        point_bag_gt = point_bag_gts[0]
        num_gts = point_bag_gt.shape[0]
        num_per_lvl = self.feature_extractor.num_sample_per_lvl()

        res = []
        for point_bag_gt in point_bag_gts:
            left_shape = point_bag_gt.shape[2:]
            point_bag_gt_outs = point_bag_gt.reshape(num_gts, -1, num_per_lvl, *left_shape)
            point_bag = point_bag_gt_outs[:, :, :-1].reshape(num_gts, -1, *left_shape)
            point_gt = point_bag_gt_outs[:, :, -1]
            res.append((point_bag, point_gt))
        if len(point_bag_gts) == 1:
            return res[0]
        return res

    def get_refine_point(self, bag_gt_outs, chosen_pts_all, gt_points, gt_labels, img_metas, gt_true_bboxes):
        """
        Args:
            point_bag_gt_outs: shape=(num_gts_all_img, num_bag_all_level, cls_out)
            chosen_pts_all: shape=(num_gts_all_img, num_bag_all_level, 2+2), 2+2 is (x, y, lvl, valid)
            gt_points: shape=[B, (num_gts, 2)]
            gt_labels: shape=[B, (num_gts, )]
        Returns:
        """
        # chosen_pts_all, the_gt_points = self.split_bag_gt(chosen_pts_all)

        all_gt_labels = torch.cat(gt_labels)
        all_gt_points = torch.cat(gt_points)
        all_gt_idx = torch.arange(len(all_gt_labels))
        _, num_all_gts, _, num_cls_out = bag_gt_outs.shape
        # assert (the_gt_points.mean(dim=1)[:, :2] == all_gt_points).all()

        lvl_max_prob_idx, bag_gt_outs_lvl = self.chose_fpn_lvl(bag_gt_outs, all_gt_labels)  #

        # step 2: aggressive all point belong to the chosen fpn level for each object/gt_point
        point_prob_cls = bag_gt_outs_lvl[0].softmax(dim=-1)  # (num_all_gts, num_per_lvl, num_cls_out)
        point_prob_ins = bag_gt_outs_lvl[1].softmax(dim=-2)
        point_prob = (point_prob_cls * point_prob_ins)[all_gt_idx, :, all_gt_labels]  # (num_all_gts, -1)

        num_per_lvl = self.feature_extractor.num_sample_per_lvl()
        chosen_pts_all = chosen_pts_all.reshape(num_all_gts, -1, num_per_lvl, 4)
        chosen_pts_lvl = chosen_pts_all[all_gt_idx, lvl_max_prob_idx]  # (num_all_gt, num_bag, 4)

        point_prob = chosen_pts_lvl[..., -1] * point_prob
        chosen_pts_lvl = chosen_pts_lvl[..., :2]

        # choose points which score > gt_score to aggressive
        point_prob = point_prob ** 0.5
        valid = (point_prob >= 0.2 * point_prob[:, -1:]) & (point_prob >= 0.1)
        point_weight = valid.float() * point_prob
        point_weight = (point_weight / point_weight.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        refine_points = (point_weight * chosen_pts_lvl).sum(dim=-2)  # (num_all_gt, 2)
        points_score = point_prob.max(dim=-1)[0]
        not_refine = points_score < 0.4
        refine_points[not_refine] = all_gt_points[not_refine]
        # points_score = point_prob_cls.mean(dim=1)

        def map_back(x):
            fmt_refine_points = []
            start_i = 0
            for gts in gt_points:
                fmt_refine_points.append(x[start_i:start_i + len(gts)])
                start_i += len(gts)
            return fmt_refine_points

        # => [B, (num_gt, )]
        fmt_refine_points = map_back(refine_points)
        fmt_points_score = map_back(points_score)
        chosen_pts_lvl = [chosen_pts_lvl[i][valid[i]] for i in range(len(point_prob))]
        chosen_pts_lvl = map_back(chosen_pts_lvl)
        lvl_max_prob_idx = map_back(lvl_max_prob_idx)
        not_refine = map_back(not_refine)

        TestCPRHead.test_refine_point(fmt_refine_points, lvl_max_prob_idx, chosen_pts_lvl,
                                      gt_points, gt_labels, img_metas, gt_true_bboxes, not_refine, fmt_points_score)
        return fmt_refine_points, fmt_points_score, lvl_max_prob_idx

    def get_refine_point2(self, point_bag_gt_outs, chosen_pts_all, gt_points, gt_labels, img_metas, gt_true_bboxes):
        """
        Args:
            point_bag_gt_outs: shape=(num_gts_all_img, num_bag_all_level, cls_out)
            chosen_pts_all: shape=(num_gts_all_img, num_bag_all_level, 2+1), 2+1 is (x, y, lvl)
            gt_points: shape=[B, (num_gts, 2)]
            gt_labels: shape=[B, (num_gts, )]
        Returns:
        """
        all_gt_labels = torch.cat(gt_labels)
        all_gt_points = torch.cat(gt_points)
        all_gt_idx = torch.arange(len(all_gt_labels))
        _, num_all_gts, _, num_cls_out = point_bag_gt_outs.shape

        prob_cls = point_bag_gt_outs[0].softmax(dim=-1)
        prob_ins = point_bag_gt_outs[1].softmax(dim=-2)
        point_prob = prob_cls * prob_ins  # (num_all_gts, -1, cls_out)

        # not_refine = point_prob.sum(dim=1).argmax(dim=-1) != all_gt_labels  # (num_gt)

        point_prob = point_prob[all_gt_idx, :, all_gt_labels]  # (num_all_gts, -1)

        _, idx = torch.topk(point_prob, 5, largest=True)
        point_prob = torch.stack([point_prob[i][idx[i]] for i in range(len(idx))])
        chosen_pts_all = torch.stack([chosen_pts_all[i][idx[i], :2] for i in range(len(idx))])  # (num_all_gts, 5, 2)

        # choose points which score > 0.5 * gt_score to aggressive
        point_prob = point_prob ** 0.5
        point_weight = point_prob
        point_weight = (point_weight / point_weight.sum(dim=1, keepdim=True)).unsqueeze(-1)
        refine_points = (point_weight * chosen_pts_all).sum(dim=-2)  # (num_all_gt, 2)
        refine_points = chosen_pts_all.mean(dim=-2)
        points_score = point_prob.mean(dim=1)

        not_refine = point_prob.max(dim=-1)[0] < 0.1
        refine_points[not_refine] = all_gt_points[not_refine]

        def map_back(x):
            fmt_refine_points = []
            start_i = 0
            for gts in gt_points:
                fmt_refine_points.append(x[start_i:start_i + len(gts)])
                start_i += len(gts)
            return fmt_refine_points

        # => [B, (num_gt, )]
        fmt_refine_points = map_back(refine_points)
        fmt_points_score = map_back(points_score)
        chosen_pts_all = map_back(chosen_pts_all)
        not_refine = map_back(not_refine)

        TestCPRHead.test_refine_point(fmt_refine_points, None, chosen_pts_all,
                                      gt_points, gt_labels, img_metas, gt_true_bboxes, not_refine)
        return fmt_refine_points, fmt_points_score, None

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

    def get_bboxes(self, cls_feat, ins_feat, img_metas,
                   cfg=None, rescale=False, with_nms=True,
                   gt_bboxes=None, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None, gt_anns_id=None):
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None

        chosen_pts_all, bag_gt_outs, neg_outs, neg_valid = self.get_cls_outs(
            cls_feat, ins_feat, gt_points, img_metas, gt_points_ignore)

        # TestCPRHead.DO_TEST = True
        points, scores, points_lvl = self.get_refine_point(bag_gt_outs, chosen_pts_all, gt_points, gt_labels,
                                                           img_metas, gt_true_bboxes)
        # TestCPRHead.DO_TEST = False

        assert sum([len(l) for l in gt_labels]) == sum([len(p) for p in points])
        final_det_bboxes = []
        det_bboxes = self.center_to_pseudo_bbox(points)
        for im_id, bboxes in enumerate(det_bboxes):
            if rescale:
                scale_factor = img_metas[im_id]['scale_factor']
                bboxes /= bboxes.new_tensor(scale_factor)
            scores_img = scores[im_id].unsqueeze(dim=-1)
            anns_id = gt_anns_id[im_id].unsqueeze(dim=-1).type_as(scores_img)
            final_det_bboxes.append(torch.cat((bboxes, scores_img, anns_id), dim=-1))
        if with_nms:
            return list(zip(final_det_bboxes, gt_labels))
        else:
            raise NotImplementedError

    def get_targets(self):
        pass


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


sta = Statistic()


class TestCPRHead(object):
    DO_TEST = False

    @staticmethod
    def test_extract_point_feat(pts_in_feat, feat, pts_feats):
        if not TestCPRHead.DO_TEST:
            return
        pts_in_feat = pts_in_feat.reshape(-1, 2)
        x, y = pts_in_feat[:, 0], pts_in_feat[:, 1]
        idx_pts_feat = feat[0][:, y.long(), x.long()].permute(1, 0)
        pts_feats = pts_feats.reshape(-1, pts_feats.shape[-1])
        s = (idx_pts_feat - pts_feats).abs().max()
        if s > 1e-4:
            print("[test_extract_point_feat]:", s, x, y, feat.shape)

    @staticmethod
    def test_chosen_fpn_level(prob_bag_cls, prob_bag_ins, lvl_max_prob_idx,
                              all_gt_idx, all_gt_labels, num_all_gts, num_level):
        if not TestCPRHead.DO_TEST:
            return
        if num_level == 1:
            return

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
                chose_lvl_d.mean().item()/other_lvl_d.mean().item(),
                (chose_lvl_d.max() / chose_lvl_d.mean()).item()) +
                tuple(d.mean(dim=(0, 2)).detach().cpu().numpy().tolist())))

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

    @staticmethod
    def test_refine_point2(points, gt_true_bboxes, not_refine):
        points, gt_true_bboxes, not_refine = torch.cat(points), torch.cat(gt_true_bboxes), torch.cat(not_refine)
        inside = (gt_true_bboxes[:, 0] < points[:, 0]) & (points[:, 0] < gt_true_bboxes[:, 2]) & \
                 (gt_true_bboxes[:, 1] < points[:, 1]) & (points[:, 1] < gt_true_bboxes[:, 3])
        outside = inside.logical_not()
        outside_bboxes = gt_true_bboxes[outside]
        outside_size = ((outside_bboxes[:, 2] - outside_bboxes[:, 0]) * (outside_bboxes[:, 3] - outside_bboxes[:, 1])) ** 0.5
        print("outside size", sta.mean("outside size", outside_size).item())
        print("outside rate", sta.mean("outside rate", outside.float()).item())
        print("refine rate", sta.mean("refine rate", not_refine.logical_not().float()).item())

    count = 0
    @staticmethod
    def test_refine_point(points, chosen_lvl, chosen_pts_all, gt_points, gt_labels, img_metas, gt_true_bboxes,
                          not_refine, fmt_points_score):
        if not TestCPRHead.DO_TEST:
            return
        TestCPRHead.count += 1
        if TestCPRHead.count > 10:
            exit(-1)
        def to_numpy(data):
            data = data[0]
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data[0], torch.Tensor):
                return [d.detach().cpu().numpy() for d in data]

        print()
        TestCPRHead.test_refine_point2(points, gt_true_bboxes, not_refine)

        img_path = img_metas[0]['filename']
        # print(img_metas[0])
        img_gt_points = to_numpy(gt_points)
        img_points = to_numpy(points)
        img_gt_labels = to_numpy(gt_labels)
        img_true_bboxes = to_numpy(gt_true_bboxes) if gt_true_bboxes is not None else None
        img_chosen_pts = to_numpy(chosen_pts_all)
        img_chosen_lvl = to_numpy(chosen_lvl) if chosen_lvl is not None else None
        img_not_refine = to_numpy(not_refine) if not_refine is not None else None
        img_scores = to_numpy(fmt_points_score) if fmt_points_score is not None else None

        if img_true_bboxes is not None:
            assert len(img_true_bboxes) == len(img_gt_points), f"{len(img_true_bboxes)} vs {len(img_gt_points)}"

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
        plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=40, c='g')
        for i in range(len(img_points)):
            # draw true bbox
            if img_true_bboxes is not None:
                draw_a_bbox(img_true_bboxes[i], color=colors[img_gt_labels[i]])
            # link gt_pt -- refine_pt
            plt.plot([img_points[i, 0], img_gt_points[i, 0]], [img_points[i, 1], img_gt_points[i, 1]], '--',
                     linewidth=3, color=colors[img_gt_labels[i]])
            if img_not_refine is None or not img_not_refine[i]:
                # chosen_pts
                plt.scatter(img_chosen_pts[i][:, 0], img_chosen_pts[i][:, 1], s=20, c='b')
                # refine_pt -- chosen_pts
                for j in range(len(img_chosen_pts[i])):
                    p1, p2 = img_gt_points[i], img_chosen_pts[i][j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[img_gt_labels[i]])
            # chosen fpn level
            if img_chosen_lvl is not None:
                plt.text(img_points[i][0], img_points[i][1], s=f"{img_chosen_lvl[i]}", fontsize=10)
            if img_scores is not None:
                plt.text(img_points[i][0], img_points[i][1], s=f"{(img_scores[i]*100).round(2)}", color=(1, 1, 1), fontsize=10)
        plt.scatter(img_gt_points[:, 0], img_gt_points[:, 1], s=40, c='g')
        plt.scatter(img_points[:, 0], img_points[:, 1], s=40, c='r')
        img_name = os.path.split(img_path)[-1]
        plt.savefig("exp/debug/CPR/vis_{}".format(img_name))
        plt.show()


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
    grid_map_func = lambda xy, wh: 2*xy / (wh-1) - 1
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
