import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class P2PHead(AnchorFreeHead):
    def __init__(self, num_classes, in_channels,
                 # loss_cls=dict(
                 #     type='FocalLoss',
                 #     use_sigmoid=True,
                 #     gamma=2.0,
                 #     alpha=0.25,
                 #     loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
                 point_anchor=[(-0.5, -0.5), (0.5, -0.5), (-0.5, -0.5), (0.5, 0.5)],
                 pts_gamma=100,
                 loss_reg=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),  # TODO => L2
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.point_anchor = torch.FloatTensor(point_anchor)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.num_cls_out = num_classes
        else:
            self.num_cls_out = num_classes + 1
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, init_cfg=init_cfg, **kwargs)

        self.pts_gamma = pts_gamma
        self.loss_reg = build_loss(loss_reg)

    def _init_layers(self):
        """ build head architecture
        Returns:
        """
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias
                )
            )

        self.conv_cls = nn.Conv2d(self.feat_channels, len(self.point_anchor) * self.num_cls_out, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, len(self.point_anchor) * 2, 3, padding=1)

    def forward(self, feats):
        """ forward head with feats of backbone output
        Args:
            feats:
        Returns:
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        cls_feat = feat
        pts_feat = feat
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        cls_out = self.conv_cls(cls_feat)
        pts_out = self.conv_reg(pts_feat)
        return cls_out, pts_out

    def _get_target_single(self,
                           anchor_pts,
                           valid_flags,
                           cls_outs,
                           gt_bboxes,
                           gt_labels,
                           # label_weights,
                           img_metas,
                           gt_points_ignore=None,
                           unmap_outputs=True):
        """
        Args:
            anchor_pts: (lvl*w*h, k, 3)
            valid_flags: (lvl*w*h, k)
            cls_outs: (lvl*w*h, k, num_cls_out) TODO: k >>> init_layer
            gt_bboxes: (num_gt, 4)
            gt_labels: (num_gt)
            img_metas: dict
            gt_bboxes_ignore:
            unmap_outputs:

        Returns:
        """
        inside_flags = valid_flags.reshape(-1)  # lvl*w*h*k
        if not inside_flags.any():
            return (None,) * 3

        anchor_pts = (anchor_pts.reshape(-1, 3))[:, :2]  # (lvl*w*h, 2)
        cls_outs = cls_outs.reshape(-1, self.num_cls_out)  # (lvl*w*h, 80)
        # assign gt and sample proposals
        proposals = anchor_pts[inside_flags, :]
        cls_pred = cls_outs[inside_flags, :]

        # #### reshape for assign
        assign_result = self.assigner.assign(
            proposals.repeat(1, 2),  # 1)points to boxes
            cls_pred,
            gt_bboxes.repeat(1, 2),  # (gt,4),
            gt_labels,
            img_metas,
            gt_bboxes_ignore=None
        )

        sampling_result = self.sampler.sample(
            assign_result,
            proposals,
            gt_bboxes
        )

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 2])  # 4
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 2])  # 4
        labels = proposals.new_full((num_valid_proposals,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds  # 15
        neg_inds = sampling_result.neg_inds  # 305

        # pos_weight = self.train_cfg.pos_weight  # reppoints
        neg_weight = self.train_cfg.neg_weight
        if len(pos_inds) > 0:
            label_weights[pos_inds] = 1.0
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            pos_gt_bboxes = pos_gt_bboxes[:, :2]
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
        if len(neg_inds) > 0:
            if neg_weight <= 0:  # default -1
                label_weights[neg_inds] = 1.0
            else:
                label_weights[neg_inds] = neg_weight

        # map up to original set of proposals 映射到
        if unmap_outputs:
            num_total_proposals = anchor_pts.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)  # (18134,4),20267,20267 >>> (20267,4)
            pos_proposals = unmap(pos_proposals, num_total_proposals,  # (data, count, inds, fill=0)
                                  inside_flags)  # (18134,4) >>> (20267,4)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return labels, label_weights, bbox_gt

    def get_targets(self,
                    anchor_pts_list,
                    valid_flag_list,
                    cls_outs_list,
                    gt_points,
                    gt_labels,
                    img_metas,
                    gt_points_ignore=None,
                    unmap_outputs=True):
        """
        Args:
            anchor_pts_list: (B, lvl*w*h, k, 3)
            valid_flag_list: (B, lvl*w*h, k)
            cls_outs_list: (B, lvl*w*h, k, num_cls_out)
            gt_points: [B, (num_gt, 4)]
            gt_labels: [B, (num_gt)]
            img_metas: [dict]
            gt_points_ignore:
            unmap_outputs:

        Returns:[B,(1280)],[B,(1280)],[B,(1280,2)]
        """
        num_imgs = len(img_metas)
        assert len(anchor_pts_list) == num_imgs == len(valid_flag_list)

        all_labels, all_label_weights, all_bbox_gt = multi_apply(
            self._get_target_single,
            anchor_pts_list,
            valid_flag_list,
            cls_outs_list,
            gt_points,
            gt_labels,
            img_metas,
            gt_points_ignore=gt_points_ignore,
            unmap_outputs=True)
        return all_labels, all_label_weights, all_bbox_gt

    def loss(self, cls_outs, pts_outs,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """ calculate loss with forward output and ground truth
        Args:
            cls_outs: [num_level, (B, num_cls_out, H, W)]
            pts_outs: [num_level, (B, len(self.point_anchor)*2, H, W)]
            gt_bboxes: [B, (num_gt, 4)]
            gt_labels: [B, (num_gt,)]
            img_metas: list[dict]
            gt_bboxes_ignore:
        Returns:
        """
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore)

        # ## reference points
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_outs]
        assert len(featmap_sizes) == len(self.point_generators)  # 1 == 1
        device = cls_outs[0].device
        # center_list: [B, num_lvl, (num_pts, 3)]
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)
        # [B, num_lvl, (num_pts, 3)] => (B, num_pts_all_lvl, 3)
        center_list = torch.stack([torch.cat(centers) for im_id, centers in enumerate(center_list)])
        valid_flags = torch.stack([torch.cat(valid) for im_id, valid in enumerate(valid_flag_list)])

        # [num_level, (B, k*num_cls_out, H, W)] => [num_level, (B, H*W, K*num_cls_out)]
        # => (B, num_pts_all_lvl, k*num_cls_out)
        k = len(self.point_anchor)
        cls_outs = torch.cat([cls_out.reshape(*cls_out.shape[:2], -1).permute(0, 2, 1)
                              for lvl, cls_out in enumerate(cls_outs)], dim=1)
        pts_outs = torch.cat([pts_out.reshape(*pts_out.shape[:2], -1).permute(0, 2, 1)
                              for lvl, pts_out in enumerate(pts_outs)], dim=1)
        cls_outs = cls_outs.reshape(*cls_outs.shape[:2], k, self.num_cls_out)
        pts_outs = pts_outs.reshape(*pts_outs.shape[:2], k, 2)

        # (B, num_pts_all_lvl, 1, 3) => (B, num_pts_all_lvl, k, 3)
        anchor_pts = center_list.unsqueeze(dim=2).repeat((1, 1, k, 1))
        valid_flags = valid_flags.unsqueeze(dim=2).repeat((1, 1, k))
        anchor_pts[..., :2] += self.point_anchor.to(anchor_pts.device)
        # (B, num_pts_all_lvl, k, 2)
        pred_pts = anchor_pts[..., :2] + pts_outs * anchor_pts[..., -1:] * self.pts_gamma

        pred_pts = torch.cat([pred_pts, anchor_pts[..., -1:]], dim=-1)
        cls_scores = cls_outs.sigmoid()

        self.get_targets(pred_pts, valid_flags, cls_scores, gt_points, gt_labels, img_metas, gt_points_ignore)

        cls_scores = cls_scores.contiguous()  # 重新开辟了一块内存,保证Tensor是contiguous的
        loss_pts = self.loss_reg(
            pts_coordinate,  # pts_out
            gt_points,
        )

        loss_dict_all = {
            'loss_cls': loss_cls,
            'loss_pts': loss_pts
        }
        return loss_dict_all
        pass

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes. [reppoints]

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)  # 4
        num_levels = len(featmap_sizes)  # 1

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i], device)
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def get_bboxes(self, cls_out, reg_out, img_metas, cfg=None, rescale=False, with_nms=True):
        """ calculate inference output with forward output
        Args:
            cls_out:
            reg_out:
            img_metas:
            cfg:
            rescale:
            with_nms:
        Returns:
        """
        pass

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
        pseudo_wh = centers[0].new_tensor(pseudo_wh)
        return [torch.cat([center - pseudo_wh / 2, center + pseudo_wh / 2], dim=-1) for center in centers]


if __name__ == '__main__':
    p2p_head = P2PHead(80, 256)
    print(p2p_head)
