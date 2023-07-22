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


@HEADS.register_module()
class P2PHead(AnchorFreeHead):
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
                 point_anchor=[(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)],  # Grid
                 # point_anchor=[(0, 0), (0, 0), (0, 0), (0, 0)],  # Center
                 assign_before_pred=False,
                 pts_gamma=100. / 8,  # γ to 100., 8 is stride
                 reg_norm=1. / 8,  # 8 is stride
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
                 loss_reg=dict(
                     type='MSELoss',  # SmoothL1Loss # beta=1.0 / 9.0,
                     loss_weight=2e-4  # λ_2
                 ),
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
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, init_cfg=init_cfg, **kwargs)

        self.point_generators = [PointGenerator() for _ in self.strides]

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler)

        self.assign_before_pred = assign_before_pred
        self.pts_gamma = pts_gamma
        self.reg_norm = reg_norm
        self.loss_reg = build_loss(loss_reg)  # 不用super().loss_bbox

    def _init_layers(self):
        """ build head architecture
        Returns:
        """
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias)
            )
            self.reg_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.conv_bias)
            )
        self.cls_out = nn.Conv2d(self.feat_channels,
                                 self.num_cls_out * self.num_points, 3, padding=1)  # 1, 1, 0 Especially for reppoints
        # num_cls_out*k
        self.reg_out = nn.Conv2d(self.feat_channels,
                                 len(self.point_anchor) * 2, 3, padding=1)

    def forward(self, feats):
        """forward the feats of backbone output to the head.

        Args:
        Returns:
        """
        cls_outs, pts_outs = multi_apply(self.forward_single, feats)
        return cls_outs, pts_outs

    def forward_single(self, feat):
        cls_feat = feat
        pts_feat = feat
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        cls_out = self.cls_out(cls_feat)
        pts_out = self.reg_out(pts_feat)
        return cls_out, pts_out

    def get_pred_points(self, cls_outs, pts_outs, img_metas):
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_outs]
        assert len(featmap_sizes) == len(self.point_generators)  # num_lvl
        # ----------------------------------------------------
        # cls_outs, pts_out >>>   (B, lvl*w*h, k*num_cls_out)     >>> (B, lvl*w*h, k, num_cls_out)
        #                         (B, lvl*w*h, k*2)                   (B, lvl*w*h, k, 2)
        cls_outs = (torch.cat(
            [cls_out.reshape(cls_out.size(0), cls_out.size(1), -1) for cls_out in cls_outs], -1)).permute(0, 2, 1)
        pts_outs = (torch.cat(
            [pts_out.reshape(pts_out.size(0), pts_out.size(1), -1) for pts_out in pts_outs], -1)).permute(0, 2, 1)
        cls_outs = cls_outs.reshape(*cls_outs.shape[:2], self.num_points, self.num_cls_out)
        pts_outs = pts_outs.reshape(*pts_outs.shape[:2], self.num_points, 2)

        # 1.1 get reference points
        device = cls_outs[0].device
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
        cls_outs = cls_outs.reshape(cls_outs.size(0), -1, self.num_cls_out)  # (B, lvl*w*h*k, num_cls_out)
        valid_flag = valid_flag.reshape(valid_flag.size(0), -1)  # (B, lvl*w*h*k)
        anchor_pts = anchor_pts.reshape(anchor_pts.shape[0], -1, 3)
        return anchor_pts, pred_pts, valid_flag, cls_outs

    def loss(self, cls_outs, pts_outs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
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
        anchor_pts, pred_pts, valid_flag, cls_outs = self.get_pred_points(cls_outs, pts_outs, img_metas)
        # assign and sample
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)  # [B, (num_gt, 2)]
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore)  # [B, (num_gt_ig, 4)]
        proposal = anchor_pts if self.assign_before_pred else pred_pts
        pro_gt_labels, pro_label_weights, pro_gt_points, pro_gt_weights = self.get_targets(
            proposal[..., :2],  # (B, lvl*w*h*k, 2)#anchor_pts,  # (B, lvl*w*h, k, 3) >>>
            valid_flag,
            cls_outs,
            gt_points, gt_labels, img_metas, gt_points_ignore
        )  # [B, (lvl*w*h*k)],[B, (lvl*w*h*k, 2)]

        # compute loss
        TestP2PHead.test_assign(img_metas, anchor_pts[..., :2], pro_gt_points, pro_gt_weights)

        num_total = sum([len(p) for p in pro_gt_labels])
        num_total_pos = sum([(l[..., 0] > 0).sum() for l in pro_gt_weights])
        loss_cls, loss_pts = multi_apply(
            self.loss_single,
            cls_outs,
            pro_gt_labels,
            pro_label_weights,
            pred_pts,
            pro_gt_points,
            pro_gt_weights,
            num_total=num_total,
            num_total_pos=num_total_pos
        )

        loss_dict_all = {
            'loss_cls': loss_cls,
            'loss_pts': loss_pts
        }
        return loss_dict_all

    def loss_single(self, cls_scores, pro_gt_labels, pro_label_weights,
                    pred_pts, pro_gt_points, pro_gt_weights,
                    num_total, num_total_pos):
        cls_avg_factor = num_total_pos
        if self.loss_cls_type == 'CrossEntropyLoss':
            cls_avg_factor = num_total
        cls_scores = cls_scores.contiguous()
        loss_cls = self.loss_cls(
            cls_scores,
            pro_gt_labels,
            pro_label_weights,
            avg_factor=cls_avg_factor
        )

        stride = pred_pts[..., -1:]
        loss_pts = self.loss_reg(
            pred_pts[..., :2] / stride / self.reg_norm,  # pts_out # (B, num_pts_all_lvl*k, 2)
            pro_gt_points / stride / self.reg_norm,
            pro_gt_weights,
            avg_factor=num_total_pos
        )

        # loss_pts = self.loss_reg(
        #     pred_pts,  # pts_out # (B, num_pts_all_lvl*k, 2)
        #     pro_gt_points,
        #     pro_gt_weights,
        #     avg_factor=num_total_pos
        # )
        return loss_cls, loss_pts

    def get_targets(self, pred_pts, valid_flag_list, cls_outs_list, gt_points, gt_labels, # label_weights,
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
        assert len(pred_pts) == num_imgs == len(valid_flag_list)

        all_labels, all_label_weights, all_bbox_gt, all_proposals_weights = multi_apply(
            self._get_target_single,
            pred_pts, valid_flag_list, cls_outs_list,
            gt_points, gt_labels, img_metas, gt_points_ignore=gt_points_ignore,
            unmap_outputs=True)
        return all_labels, all_label_weights, all_bbox_gt, all_proposals_weights

    def _get_target_single(self, pred_pts, valid_flags, cls_outs, gt_points, gt_labels, img_metas, gt_points_ignore=None,
                           unmap_outputs=True):
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
        inside_flags = valid_flags  # lvl*w*h*k [0]
        if not inside_flags.any():
            return (None,) * 3

        proposals = pred_pts[inside_flags, :]
        cls_pred = cls_outs[inside_flags, :]

        assign_result = self.assigner.assign(proposals, cls_pred, gt_points, gt_labels, img_metas, gt_bboxes_ignore=None)
        sampling_result = self.sampler.sample(assign_result, proposals, gt_points)
        labels, label_weights, bbox_gt, proposals_weights = self.sample_result_to_target(proposals, sampling_result,
                                                                                         gt_labels)
        # map up to original set of proposals 映射到
        if unmap_outputs:
            do_unmap = partial(unmap, count=pred_pts.size(0), inds=inside_flags)    # lvl*w*h*k
            labels, label_weights, bbox_gt, proposals_weights = map(do_unmap, [labels, label_weights,
                                                                               bbox_gt, proposals_weights])
        return labels, label_weights, bbox_gt, proposals_weights

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

    def get_bboxes(self, cls_outs, pts_outs, img_metas, cfg=None, rescale=False, with_nms=True):
        assert len(cls_outs) == len(pts_outs)
        anchor_pts, pred_pts, valid_flag, cls_outs = self.get_pred_points(cls_outs, pts_outs, img_metas)
        # (B, num_pts_all_lvl * k, 2),(B, lvl * w * h*k),(B, lvl * w * h * k, num_cls_out)
        result_list = []
        for img_id in range(len(img_metas)):
            # ##
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            point_scores, labels = self._get_bboxes_single(pred_pts[img_id][..., :2], valid_flag[img_id], cls_outs[img_id],
                                                           img_shape, scale_factor, cfg, rescale)
            bbox_scores = self.center_to_pseudo_bbox([point_scores])[0]
            result_list.append((bbox_scores, labels))
        return result_list

    def _get_bboxes_single(self, pred_pts, valid_flag, cls_outs, img_shape, scale_factor, cfg, rescale=False, with_nms=True):
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
        for i_lvl, (cls_score, points_pred) in enumerate(zip(cls_outs, pred_pts)):    # (w*h*k,num_cls_out),(w*h*k,2)
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
            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]

            ### after nms
            #_bboxes (*5)
            _bboxes = bbox_outputs[0]
            _labels = bbox_outputs[1]
            _scores = _bboxes.new_full((_bboxes.shape[0],self.num_classes),0)
            _scores[torch.arange(_bboxes.shape[0]),_labels] = _bboxes[:,4]
            aug_bboxes.append(_bboxes[:,:4])
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
        print(img_metas[0])
        all_proposal_points = to_numpy(proposal_points)
        valid = (proposal_weight[0][:, 0] > 0) & (proposal_weight[0][:, 1] > 0)
        proposal_points = proposal_points[0][valid]
        matched_gt_points = matched_gt_points[0][valid]
        proposal_points = to_numpy([proposal_points])
        matched_gt_points = to_numpy([matched_gt_points])

        from PIL import Image
        import matplotlib.pyplot as plt
        from ssdcv.vis.visualize import get_hsv_colors, draw_a_bbox
        from ssdcv.plot_paper.plt_paper_config import set_plt
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


if __name__ == '__main__':
    head = P2PHead(256, 80)
    print(head)
