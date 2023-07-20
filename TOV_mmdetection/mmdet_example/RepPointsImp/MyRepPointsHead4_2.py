"""
point_generator()
assigner
sampler()

centers:(B, num_levels, (H*W, 3))  # 3 is (x, y, stride)
    get_points:
        point_generator.grid_points
        point_generator.valid_flags
get_targets: () <- centers, gt_bboxes,
    flat_proposals(cat levels): (B, (N, 3/4))
    for proposals(N, 3/4) in each image:
        assigner: AssignResult
            num_gts: int, number of gts
            gt_inds: matched gt idx + 1, 0 for no match, shape=(N,)
            max_overlaps:
            labels: matched gt's class label, -1 for no match(Background), shape=(N,)
        sampler:
            pos_inds/pos_bboxes: sampled positive indexes/bboxes of proposal
            neg_inds/neg_bboxes: sampled negative indexes/bboxes of proposal
            pos_is_gt: pos_is_gt[i] means whether pos_bboxes[i] is gt or not
            pos_assigned_gt_inds: matched gt idx of sampled pos proposal, start from 0
            pos_gt_bboxes/pos_gt_labels: matched gt's bbox/label of sampled pos proposal
        sample_result_to_targets:
            labels: shape=(N,), num_class for Background <- pos_inds, pos_assigned_gt_inds / pos_gt_labels
            labels_weight: shape=(N,), <- pos_inds, neg_inds
            pos_gt_bboxes, pos_proposals, pos_proposals_weight: shape=(N, X) <- pos_inds, pos_gt_bboxes | pos_bboxes | 1
            num_pos, num_neg:
"""


from MyRepPointsHead3 import MyRepPointsHead3
from Data import gen_pseudo_inputs
from AssignerSmapler import MyPointAssigner, MyPointGenerator

from mmdet.models.builder import build_loss
from mmdet.core import PointGenerator, multi_apply
import numpy as np
import torch
from mmdet.core import build_assigner, build_sampler


class MyRepPointsHead4(MyRepPointsHead3):
    def __init__(self, num_classes, in_channels,
                 point_strides=[8, 16, 32, 64, 128],
                 loss_cls=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0,
                               use_sigmoid=True),
                 loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),  # ***************** star 2
                 loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 *args, **kwargs):
        super().__init__(num_classes, in_channels, *args, **kwargs)

        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            else:
                self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             label_channels=1,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
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
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds)

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

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

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             stage=stage,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg)

    def offset_to_pts(self, pts_preds, center_list):
        """
        Args:
            pts_preds: shape is (num_levels, (B, C, H, W)), C means (y0, x0, y0, x1, ...)
            center_list: shape is (B, num_levels, (H*W, 3)), 3 means (x, y, stride)
        Returns:
            pts_xy_preds: shape is (B, num_levels, (H*W, num_points*2)), num_points*2 means (x0, y0, x1, y1, x2, y2 ...)
        """
        for img_id, center in enumerate(center_list):
            for lvl in range(len(center)):
                center = center[:, :2].repeat(1, self.num_points)
                pts_xy = pts_preds[lvl][img_id].permute((1, 2, 0))
                pts_xy = pts_xy.view((-1, pts_xy.shape[-1]))
                pts_y, pts_x = pts_xy[:, 0::2], pts_xy[:, 1::2]
                torch.stack([pts_x, pts_y], dim=-1)

    def loss(self, cls_scores, pts_preds_init, pts_preds_refine,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):


        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # center_list: (B, num_levels, (H * W, 3))  # 3 is (x, y, stride)
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)


