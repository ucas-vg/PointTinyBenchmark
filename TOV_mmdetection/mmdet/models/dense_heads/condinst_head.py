import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import BaseModule, force_fp32
from mmcv.image import tensor2imgs
from skimage import color

from mmdet.core import distance2bbox, multi_apply, nms_with_others, reduce_mean
from .anchor_free_head import AnchorFreeHead
from ..builder import HEADS, build_loss
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import bbox2result

INF = 1e8


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # this equation is equal to log(p_i * p_j + (1 - p_i) * (1 - p_j))
    # max is used to prevent overflow
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    """
    Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
    :param x:
    :param target:
    :return:
    """
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def compute_project_term(mask_scores, gt_bitmasks, weight, gt_bitmasks_weights, ins_scores=None):
    # mask_scores=mask_scores*gt_bitmasks_weights
    # gt_bitmasks=gt_bitmasks*gt_bitmasks_weights
    if ins_scores is not None:
        ins_scores_x = (ins_scores * gt_bitmasks).softmax(dim=-2)
        ins_scores_y = (ins_scores * gt_bitmasks).softmax(dim=-1)

        mask_losses_y = dice_coefficient(
            (mask_scores * ins_scores_x).sum(dim=2, keepdim=True),
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            (mask_scores * ins_scores_y).sum(dim=3, keepdim=True),
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
    else:
        mask_scores = mask_scores * gt_bitmasks_weights
        gt_bitmasks = gt_bitmasks * gt_bitmasks_weights
        mask_losses_y = dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )

        # mask_scores = mask_scores * gt_bitmasks_weights
        # mask_scores2 = mask_scores * (1 - gt_bitmasks_weights)
        # mask_losses_y_mean = dice_coefficient(
        #     mask_scores2.mean(dim=2, keepdim=True),
        #     gt_bitmasks.mean(dim=2, keepdim=True)
        # )
        # mask_losses_x_mean = dice_coefficient(
        #     mask_scores2.mean(dim=3, keepdim=True),
        #     gt_bitmasks.mean(dim=3, keepdim=True)
        # )
    return (mask_losses_x * weight + mask_losses_y * weight).mean()
    # + (            mask_losses_x_mean * weight + mask_losses_y_mean * weight).mean()


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert isinstance(factor, int)

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]


def get_original_image(img, img_meta):
    """

    :param img(Tensor):  the image with pading [3, h, w]
    :param img_meta(dict): information about the image
    :return: original_img(Tensor)
    """
    original_shape = img_meta["img_shape"]
    original_shape_img = img[:, :original_shape[0], :original_shape[1]]
    img_norm_cfg = img_meta["img_norm_cfg"]
    original_img = tensor2imgs(original_shape_img.unsqueeze(0).float(), mean=img_norm_cfg["mean"], std=img_norm_cfg["std"],
                               to_rgb=img_norm_cfg["to_rgb"])[0]  # in BGR format
    original_img = torch.tensor(original_img[:, :, ::-1].copy()).permute(2, 0, 1)  # to RGB tensor [c h w]
    original_img = original_img.float().to(img.device)

    return original_img

    # cv2.imwrite("show/cv_{}".format(img_meta["filename"].split("/")[-1]), original_img)
    # Image.fromarray(original_img).save("show/pil_{}".format(img_meta["filename"].split("/")[-1]))


def unfold_wo_center(x, kernel_size, dilation):
    """
    :param x: [N, C, H, W]
    :param kernel_size: k
    :param dilation:
    :return: [N, C, K^2-1, H, W]
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


def get_image_color_similarity(image, mask, pairwise_size, pairwise_dilation):
    """
    \
    :param self:
    :param image: [1, 3, H, W]
    :param mask: [H, W]
    :param pairwise_size: k
    :param pairwise_dilation: d
    :return:[1, 8, H, W]
    """
    assert image.dim() == 4
    assert image.size(0) == 1

    unfolded_image = unfold_wo_center(
        image, kernel_size=pairwise_size, dilation=pairwise_dilation
    )

    diff = image.unsqueeze(2) - unfolded_image

    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weight = unfold_wo_center(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=pairwise_size, dilation=pairwise_dilation
    )[:, 0, :, :, :]

    return similarity * unfolded_weight


def kernel_and_pad(number):
    number = torch.ceil(number)
    if number % 2 == 0:
        kernel = int(number - 1)
        pad = int((number - 2) / 2)
    else:
        kernel = int(number)
        pad = int((number - 1) / 2)

    return kernel, pad


@HEADS.register_module()
class CondInstBoxHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 16), (16, 32), (32, 64), (64, 128),
                                 (128, INF)),
                 # regress_ranges=((-1, INF),),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
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
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats, top_module):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            top_module (nn.Module): Generate dynamic parameters from FCOS
                regression branch.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                param_preds (list[Tensor]): dynamic parameters generated from \
                    each scale level, each is a 4-D-tensor, the channel number \
                    is decided by top_module.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, top_module=top_module)

    def forward_single(self, x, scale, stride, top_module):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
            top_module (nn.Module): Exteral input module. #---------------

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(CondInstBoxHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        param_pred = top_module(reg_feat)
        return cls_score, bbox_pred, centerness, param_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, gt_inds = \
            self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        flatten_gt_inds = torch.cat(gt_inds)
        flatten_img_inds = []
        flatten_level_inds = []
        for i, featmap_size in enumerate(featmap_sizes):
            H, W = featmap_size
            img_inds = torch.arange(num_imgs, device=bbox_preds[0].device)
            flatten_img_inds.append(img_inds.repeat_interleave(H * W))
            flatten_level_inds.append(torch.full(
                (num_imgs * H * W,), i, device=bbox_preds[0].device).long())
        flatten_img_inds = torch.cat(flatten_img_inds)
        flatten_level_inds = torch.cat(flatten_level_inds)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)
        return (losses, flatten_points, flatten_level_inds, flatten_img_inds,
                flatten_gt_inds)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        cum = 0
        for gt_inds, gt_bboxes in zip(gt_inds_list, gt_bboxes_list):
            gt_inds[gt_inds != -1] += cum
            cum += gt_bboxes.size(0)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        gt_inds_list = [gt_inds.split(num_points, 0) for gt_inds in gt_inds_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_gt_inds = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_gt_inds.append(
                torch.cat([gt_inds[i] for gt_inds in gt_inds_list]))
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_gt_inds)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        min_area_inds[min_area == INF] = -1

        return labels, bbox_targets, min_area_inds

    def simple_test(self, feats, top_module, img_metas, rescale=False):
        outs = self.forward(feats, top_module)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   param_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, param_preds,
                                       mlvl_points, img_shapes, scale_factors,
                                       cfg, rescale, with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    param_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_coors = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_param_pred = []

        for cls_score, bbox_pred, centerness, param_pred, points in zip(
                cls_scores, bbox_preds, centernesses, param_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            param_num = param_pred.size(1)
            param_pred = param_pred.permute(0, 2, 3, 1).reshape(batch_size,
                                                                -1, param_num)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    raise NotImplementedError("CondInst doesn't support ONNX currently")
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]
                    param_pred = param_pred[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_coors.append(points)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_param_pred.append(param_pred)

        batch_lvl_inds = torch.cat(
            [torch.full_like(ctr, i).long()
             for i, ctr in enumerate(mlvl_centerness)], dim=1)
        batch_mlvl_coors = torch.cat(mlvl_coors, dim=1)
        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_param_pred = torch.cat(mlvl_param_pred, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            raise NotImplementedError("CondInst doesn't support ONNX currently")
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (lvl_inds, mlvl_coors, mlvl_bboxes, mlvl_scores, mlvl_centerness,
                 mlvl_param_pred) in zip(batch_lvl_inds, batch_mlvl_coors,
                                         batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness, batch_mlvl_param_pred):
                det_bbox, det_label, others = nms_with_others(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    others=[mlvl_param_pred,
                            mlvl_coors,
                            lvl_inds]
                )
                outputs = (det_bbox, det_label) + tuple(others)
                det_results.append(outputs)
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness,
                                   batch_mlvl_param_pred,
                                   batch_mlvl_coors,
                                   batch_lvl_inds)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                                         left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


@HEADS.register_module()
class CondInstSegmHead(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 in_stride=8,
                 stacked_convs=2,
                 feat_channels=128,
                 loss_segm=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     override=dict(
                         type='Kaiming',
                         name='segm_conv',
                         bias_prob=0.01))):
        super(CondInstSegmHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.loss_segm = build_loss(loss_segm)
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        segm_branch = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            segm_branch.append(ConvModule(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        self.segm_branch = nn.Sequential(*segm_branch)
        self.segm_conv = nn.Conv2d(
            self.feat_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        return self.segm_conv(self.segm_branch(x))

    @force_fp32(apply_to=('segm_pred',))
    def loss(self, segm_pred, gt_masks, gt_labels):
        semantic_targets = self.get_targets(gt_masks, gt_labels)
        semantic_targets = semantic_targets.flatten()
        num_pos = (semantic_targets != self.num_classes).sum().float()
        num_pos = num_pos.clamp(min=1.)

        segm_pred = segm_pred.permute(0, 2, 3, 1)
        segm_pred = segm_pred.flatten(end_dim=2)
        loss_segm = self.loss_segm(
            segm_pred,
            semantic_targets,
            avg_factor=num_pos)
        return dict(loss_segm=loss_segm)

    def get_targets(self, gt_masks, gt_labels):
        semantic_targets = []
        for cur_gt_masks, cur_gt_labels in zip(gt_masks, gt_labels):
            h, w = cur_gt_masks.size()[-2:]
            areas = torch.sum(cur_gt_masks, dim=(1, 2), keepdim=True)
            areas = areas.repeat(1, h, w)
            areas[cur_gt_masks == 0] = INF
            min_areas, inds = torch.min(areas, dim=0, keepdim=True)

            cur_gt_labels = cur_gt_labels[:, None, None].repeat(1, h, w)
            per_img_targets = torch.gather(cur_gt_labels, 0, inds)
            per_img_targets[min_areas == INF] = self.num_classes
            semantic_targets.append(per_img_targets)

        stride = self.in_stride
        semantic_targets = torch.cat(semantic_targets, dim=0)
        semantic_targets = semantic_targets[:, stride // 2::stride, stride // 2::stride]
        return semantic_targets


@HEADS.register_module()
class CondInstMaskBranch(BaseModule):

    def __init__(self,
                 in_channels=256,
                 in_indices=[0, 1, 2],
                 strides=[8, 16, 32],
                 branch_convs=4,
                 branch_channels=128,
                 branch_out_channels=8,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(CondInstMaskBranch, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert len(in_indices) == len(strides)
        assert in_indices[0] == 0
        self.in_indices = in_indices
        self.strides = strides
        self.branch_convs = branch_convs
        self.branch_channels = branch_channels
        self.branch_out_channels = branch_out_channels
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.refines = nn.ModuleList()
        for _ in self.in_indices:
            self.refines.append(ConvModule(
                self.in_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))

        mask_branch = []
        for _ in range(self.branch_convs):
            mask_branch.append(ConvModule(
                self.branch_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        mask_branch.append(
            nn.Conv2d(self.branch_channels, self.branch_out_channels, 1))
        self.mask_branch = nn.Sequential(*mask_branch)

    def forward(self, x):
        mask_stride = self.strides[self.in_indices[0]]
        mask_x = self.refines[0](x[self.in_indices[0]])
        for i in range(1, len(self.in_indices)):
            stride = self.strides[i]
            assert stride % mask_stride == 0
            p_x = self.refines[i](x[self.in_indices[i]])
            p_x = aligned_bilinear(p_x, stride // mask_stride)
            mask_x = mask_x + p_x
        return self.mask_branch(mask_x)


@HEADS.register_module()
class CondInstMaskHead(BaseModule):

    def __init__(self,
                 in_channels=8,
                 in_stride=8,
                 out_stride=4,
                 dynamic_convs=3,
                 dynamic_channels=8,
                 disable_rel_coors=False,
                 bbox_head_channels=256,
                 sizes_of_interest=[64, 128, 256, 512, 1024],
                 max_proposals=500,
                 topk_per_img=-1,
                 boxinst_enabled=False,
                 bottom_pixels_removed=10,
                 pairwise_size=3,
                 pairwise_dilation=2,
                 pairwise_color_thresh=0.3,
                 pairwise_warmup=10000,
                 num_classes=80,
                 merge_featmap=False,
                 soft_project=False,
                 mil_proj=False,
                 enlarge_rate=False,
                 whole_map=False,
                 with_class_embedding=False,
                 random_sample_PE=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear', ''],
                     std=0.01,
                     bias=0)):
        super(CondInstMaskHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert in_stride >= out_stride
        assert in_stride % out_stride == 0
        self.in_stride = in_stride
        self.out_stride = out_stride
        assert dynamic_channels > 1
        self.dynamic_convs = dynamic_convs
        self.dynamic_channels = dynamic_channels
        self.disable_rel_coors = disable_rel_coors
        self.random_sample_PE = random_sample_PE
        dy_weights, dy_biases = [], []
        dynamic_in_channels = in_channels if disable_rel_coors else in_channels + 2
        for i in range(dynamic_convs):
            in_chn = dynamic_in_channels if i == 0 else dynamic_channels
            out_chn = 1 if i == dynamic_convs - 1 else dynamic_channels
            dy_weights.append(in_chn * out_chn)
            dy_biases.append(out_chn)
        self.dy_weights = dy_weights
        self.dy_biases = dy_biases
        self.num_gen_params = sum(dy_weights) + sum(dy_biases)
        self.bbox_head_channels = bbox_head_channels

        self.register_buffer("sizes_of_interest", torch.tensor(sizes_of_interest))
        assert max_proposals == -1 or topk_per_img == -1, \
            'max_proposals and topk_per_img cannot be used at the same time'
        self.max_proposals = max_proposals
        self.topk_per_img = topk_per_img

        self.boxinst_enabled = boxinst_enabled
        self.bottom_pixels_removed = bottom_pixels_removed
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh

        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = pairwise_warmup

        self.iou_mask_sum = 0
        self.iou_box_sum = 0
        self.iou_num = 0

        self.norm_cfg = norm_cfg
        self.fp16_enable = False
        self.mil_proj = mil_proj
        self.enlarge_rate = enlarge_rate
        self.num_classes = num_classes
        self.with_class_embedding = with_class_embedding
        self._init_layers()
        self.merge_featmap = merge_featmap
        self.soft_project = soft_project

        self.whole_map = whole_map

        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.param_conv = nn.Conv2d(
            256,
            self.num_gen_params,
            3,
            stride=1,
            padding=1)

        self.param_linear = nn.Linear(
            self.bbox_head_channels * 2 if self.with_class_embedding else self.bbox_head_channels,
            self.num_gen_params * 2 if self.mil_proj or self.enlarge_rate else self.num_gen_params, )

        self.class_embedding = nn.Embedding(
            self.num_classes + 1, self.bbox_head_channels)

    def parse_dynamic_params(self, params):
        num_insts = params.size(0)
        params_list = list(torch.split_with_sizes(
            params, self.dy_weights + self.dy_biases, dim=1))
        weights_list = params_list[:self.dynamic_convs]
        biases_list = params_list[self.dynamic_convs:]

        for i in range(self.dynamic_convs):
            if i < self.dynamic_convs - 1:
                weights_list[i] = weights_list[i].reshape(
                    num_insts * self.dynamic_channels, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(
                    num_insts * self.dynamic_channels)
            else:
                weights_list[i] = weights_list[i].reshape(num_insts * 1, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(num_insts * 1)
        return weights_list, biases_list

    def forward(self, feat, params, coors, level_inds, img_inds):

        mask_feat = feat[img_inds]
        N, _, H, W = mask_feat.size()
        if not self.disable_rel_coors:
            shift_x = torch.arange(0, W * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y = torch.arange(0, H * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            locations = torch.stack([shift_x, shift_y], dim=0) + self.in_stride // 2
            level_inds = torch.cat(level_inds)
            soi = self.sizes_of_interest.float()[level_inds]

            if self.random_sample_PE:
                rel_coors = torch.cat(coors)[..., None, None][:, :2] - locations[None, None, ...]
                rel_coors = rel_coors.reshape(N, -1, H, W)
            else:
                rel_coors = torch.cat(coors)[..., None, None][:, :2] - locations[None]

            rel_coors = rel_coors / soi[..., None, None, None]

            ############
            # A = (torch.cat(coors)[..., None, None][:, :2] - locations[None])
            # B = torch.cat(coors)[..., None, None][:, 2:] / 4
            # rel_coors = torch.exp(-A ** 2 / (2 * B ** 2)) / B
            # rel_coors = (rel_coors.flatten(2).max(-1)[0][..., None, None] - rel_coors) * (
            #         (A > 0) * 1 - (A < 0) * 1) * 100
            ############

            ############
            # A1 = (locations[None] - torch.cat(coors)[..., None, None][:, :2] - torch.cat(coors)[..., None, None][:,
            #                                                                    2:] / 2)
            # A2 = (locations[None] - torch.cat(coors)[..., None, None][:, :2] + torch.cat(coors)[..., None, None][:,
            #                                                                    2:] / 2)
            # B = torch.cat(coors)[..., None, None][:, 2:] / 2
            # rel_coors = (A1 ** 3 + B ** 3) * (locations[None] > torch.cat(coors)[..., None, None][:, :2]) + (
            #         A2 ** 3 - B ** 3) * (locations[None] < torch.cat(coors)[..., None, None][:, :2])
            # rel_coors = rel_coors / soi[..., None, None, None] ** 3

            ############
            pass
        if self.mil_proj == True:
            params1 = [i[:, :self.num_gen_params] for i in params]
            params2 = [i[:, self.num_gen_params:] for i in params]
            weights1, biases1 = self.parse_dynamic_params(torch.cat(params1))
            weights2, biases2 = self.parse_dynamic_params(torch.cat(params2))
            x2 = x1 = torch.cat([rel_coors, mask_feat], dim=1).reshape(1, -1, H, W)
            for i, (w, b) in enumerate(zip(weights1, biases1)):
                x1 = F.conv2d(x1, w, bias=b, stride=1, padding=0, groups=N)
                if i < self.dynamic_convs - 1:
                    x1 = F.relu(x1)
            # x2 = mask_feat.reshape(1, -1, H, W)
            # x2 = mask_feat.reshape(1, -1, H, W)
            for i, (w, b) in enumerate(zip(weights2, biases2)):
                # if i == 0:
                #     w = w[:, 2:]

                x2 = F.conv2d(x2, w, bias=b, stride=1, padding=0, groups=N)
                if i < self.dynamic_convs - 1:
                    x2 = F.relu(x2)
            x = torch.cat([x1, x2], dim=0)
            if self.whole_map:
                rel_coors = rel_coors * 0.0
                mask_feat_2 = torch.cat([rel_coors, mask_feat], dim=1)
                # weights, biases = self.parse_dynamic_params(torch.cat(params))
                x2 = mask_feat_2.reshape(1, -1, H, W)
                for i, (w, b) in enumerate(zip(weights1, biases1)):
                    x2 = F.conv2d(x2, w, bias=b, stride=1, padding=0, groups=N)
                    if i < self.dynamic_convs - 1:
                        x2 = F.relu(x2)
                x = torch.cat([x, x2], dim=0)
            x = x.permute(1, 0, 2, 3)
        else:
            mask_feat_1 = torch.cat([rel_coors, mask_feat], dim=1)
            params1 = [i[:, :self.num_gen_params] for i in params]
            weights, biases = self.parse_dynamic_params(torch.cat(params1))
            x = mask_feat_1.reshape(1, -1, H, W)
            for i, (w, b) in enumerate(zip(weights, biases)):

                x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=N)
                if i < self.dynamic_convs - 1:
                    x = F.relu(x)

                    # x = F.relu(x)
            if self.enlarge_rate:
                rate = (self.enlarge_rate - 1.05) * torch.rand((N, 2), device=rel_coors.device) + 1.05
                rel_coors = rel_coors * rate[..., None, None]
                mask_feat_2 = torch.cat([rel_coors, mask_feat], dim=1)
                weights, biases = self.parse_dynamic_params(torch.cat(params1))
                x2 = mask_feat_2.reshape(1, -1, H, W)
                for i, (w, b) in enumerate(zip(weights, biases)):
                    x2 = F.conv2d(x2, w, bias=b, stride=1, padding=0, groups=N)
                    if i < self.dynamic_convs - 1:
                        x2 = F.relu(x2)
                x = torch.cat([x, x2], dim=0)
            x = x.permute(1, 0, 2, 3)
        x = aligned_bilinear(x, self.in_stride // self.out_stride)
        return x

    def training_sample(self, pseudo_boxes, gt_labels, num_levels, others, cascade_weight=None, test=None):
        num_imgs = len(pseudo_boxes)
        # num_gts = torch.cat(pseudo_boxes).shape[0]
        if self.merge_featmap and test:
            x_feat = others['x_feat']
            num_ins = x_feat[0].shape[1]
            filtered_scores = others['filtered_scores']
            filtered_boxes = others['filtered_boxes']
            pseudo_boxes = [i.reshape(-1, i.shape[-1]) for i in filtered_boxes]
            if cascade_weight is not None:
                cascade_weight = cascade_weight[:, None].expand(len(cascade_weight), num_ins).reshape(-1)
            x_feat = [i.reshape(-1, i.shape[-1]) for i in x_feat]

        else:
            x_feat = others['x_feat']
            filtered_scores = others['filtered_scores']
            weight = [w / w.sum(dim=1, keepdim=True) + 1e-8 for w in filtered_scores]
            x_feat = [(weight[i][:, :, None] * x_feat[i]).sum(1) for i in range(len(x_feat))]
        batch_gt = [len(i) for i in x_feat]
        x_feat = torch.cat(x_feat)
        #
        # #####################################
        if self.with_class_embedding:
            lb = torch.cat(gt_labels)
            class_feat = self.class_embedding.weight[lb]
            other_lb = []
            for i in range(len(gt_labels)):
                for j in gt_labels[i]:
                    if (gt_labels[i] != j).sum() == 0:
                        lb_all = torch.arange(self.num_classes).to(lb.device)
                        tochoose = torch.nonzero(lb_all != j)
                        id = torch.randint(0, len(tochoose), [1]).to(lb.device)
                        other_lb.append(lb_all[tochoose[id]])
                    else:
                        tochoose = torch.nonzero(gt_labels[i] != j)
                        id = torch.randint(0, len(tochoose), [1]).to(lb.device)
                        other_lb.append(gt_labels[i][tochoose[id]])
            other_lb = torch.tensor(other_lb).to(lb.device)
            other_class_feat = self.class_embedding.weight[other_lb]
            bg_feat = self.class_embedding.weight[[-1 for i in lb]]

            x_feat_ = torch.cat([x_feat, class_feat], dim=1)
            other_x_feat = torch.cat([x_feat, other_class_feat], dim=1)
            bg_x_feat = torch.cat([x_feat, bg_feat], dim=1)

            # ########################################3
            param_preds_list = []
            param_preds = self.param_linear(x_feat_)
            param_preds_list.append(torch.split(param_preds, batch_gt))
            other_param_preds = self.param_linear(other_x_feat)
            param_preds_list.append(torch.split(other_param_preds, batch_gt))
            bg_param_preds = self.param_linear(bg_x_feat)
            param_preds_list.append(torch.split(bg_param_preds, batch_gt))
        else:
            param_preds = self.param_linear(x_feat)
            param_preds_list = torch.split(param_preds, batch_gt)

        level_inds = self.map_roi_levels(pseudo_boxes, num_levels=num_levels)
        img_inds = []
        for idx, box in enumerate(pseudo_boxes):
            for i in box:
                img_inds.append(idx)

        coors = [bbox_xyxy_to_cxcywh(box).round() for box in pseudo_boxes]
        if self.random_sample_PE:
            pt_list = []
            for i, box in enumerate(pseudo_boxes):
                wh = bbox_xyxy_to_cxcywh(box)[:, 2:]
                random_pts = box[:, None, :2] + wh[:, None, :] * torch.rand((len(wh), self.random_sample_PE - 1, 2),
                                                                            device=box.device)
                random_pts = torch.cat([coors[i][:, None], random_pts], dim=1)
                pt_list.append(random_pts)
            coors = pt_list

        return param_preds_list, coors, level_inds, img_inds, pseudo_boxes, cascade_weight

    def get_mask(self, mask_preds, others, img_metas, test=False):
        # mask_pred = mask_preds[:, 1, None] if self.mil_proj else mask_preds[:, 0, None]
        mask_pred = mask_preds[:, 0, None]
        mask_pred = mask_pred.sigmoid().type(torch.float)

        if self.merge_featmap and test:
            num_ins = others['x_feat'][0].shape[1]
            mask_pred = mask_pred.reshape(-1, num_ins, *mask_pred.shape[2:])
            filtered_scores = others['filtered_scores']
            weight = [w / w.sum(dim=1, keepdim=True) + 1e-8 for w in filtered_scores]
            # weight=[w / w[:, 0, None] for w in weight]
            # weight=[w>(1/(2*num_ins)) for w in weight]
            # mask_pred=(torch.cat(weight)[..., None, None] * mask_pred).max(dim=1, keepdim=True)[0]
            mask_pred = (torch.cat(weight)[..., None, None] * mask_pred).sum(dim=1, keepdim=True)
            return mask_pred
        # elif self.enlarge_rate:
        #     enlarge_pred = mask_preds[:, 1, None].sigmoid().type(torch.float)
        #     map = (enlarge_pred - mask_pred.sum(0, keepdim=True)) + mask_pred
        #     mask_pred = torch.max(map, mask_pred)
        #     return mask_pred
        else:
            return mask_pred

    def mask_mean_iou(self, mask_pred, gt_masks, gt_bboxes, pseudo_boxes, img_metas):

        # mask_pred = mask_pred.sigmoid().type(torch.float)
        batch_split = [len(i) for i in gt_bboxes]
        stride = self.out_stride
        mask_pred = aligned_bilinear(mask_pred, stride).squeeze(1)
        mask_pred = (mask_pred > 0.5).type(torch.int8)
        pseudo_bboxes = self.masks_to_boxes(mask_pred)
        gt_bboxes = torch.cat(gt_bboxes)
        iou_box = bbox_overlaps(pseudo_bboxes, gt_bboxes, is_aligned=True)
        iou_sum, num, mask_iou = 0, 0, None
        pseudo_bboxes = pseudo_bboxes.split(batch_split)
        mask_pred_list = mask_pred.split(batch_split)
        if gt_masks is not None:
            gt_masks = [i.masks for i in gt_masks]
            for i, gt_mask in enumerate(gt_masks):
                gt_mask = torch.tensor(gt_mask).to(mask_pred_list[0].device)
                mask = mask_pred_list[i]
                shape = gt_mask.shape
                mask = mask[:, :shape[1], :shape[2]]
                intersection = ((mask + gt_mask) == 2).reshape(shape[0], -1).sum(1)
                union = ((mask + gt_mask) >= 1).reshape(shape[0], -1).sum(1)
                # union = (gt_mask >= 1).reshape(shape[0], -1).sum(1)
                IoU = intersection / union
                iou_sum += IoU.sum()
                num += shape[0]
                mask_iou = iou_sum / num
        return mask_iou, iou_box.mean(), pseudo_bboxes, mask_pred_list

    def masks_to_boxes(self, masks):
        """Compute the bounding boxes around the provided masks

        The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

        Returns a [N, 4] tensors, with the boxes in xyxy format
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)

        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float).to(masks.device)
        x = torch.arange(0, w, dtype=torch.float).to(masks.device)
        y, x = torch.meshgrid(y, x)

        x_mask = (masks * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        y_mask = (masks * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        return torch.stack([x_min, y_min, x_max, y_max], 1)

    def map_roi_levels(self, bboxes, num_levels, finest_scale=56):
        target_lvls_list = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            scale = torch.sqrt(
                (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]))
            target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
            target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
            target_lvls_list.append(target_lvls)
        return target_lvls_list

    # def map_roi_levels(self, bboxes, num_levels, finest_scale=56):
    #     target_lvls_list = []
    #     for i in range(len(bboxes)):
    #         bbox = bboxes[i]
    #         scale = torch.sqrt(
    #             (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]))
    #         # target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    #         # target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    #         target_lvls_list.append(scale)
    #     return target_lvls_list

    def simple_test(self,
                    mask_feat,
                    det_labels,
                    det_params,
                    det_coors,
                    det_level_inds,
                    img_metas,
                    num_classes,
                    rescale=False):
        num_imgs = len(img_metas)
        num_inst_list = [param.size(0) for param in det_params]
        det_img_inds = [
            torch.full((num,), i, dtype=torch.long, device=mask_feat.device)
            for i, num in enumerate(num_inst_list)
        ]

        det_params = torch.cat(det_params, dim=0)
        det_coors = torch.cat(det_coors, dim=0)
        det_level_inds = torch.cat(det_level_inds, dim=0)
        det_img_inds = torch.cat(det_img_inds, dim=0)
        if det_params.size(0) == 0:
            segm_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            return segm_results

        mask_preds = self.forward(mask_feat, det_params, det_coors, det_level_inds,
                                  det_img_inds)
        mask_preds = mask_preds.sigmoid()
        mask_preds = aligned_bilinear(mask_preds, self.out_stride)

        segm_results = []
        mask_preds = torch.split(mask_preds, num_inst_list, dim=0)
        for cur_mask_preds, cur_labels, img_meta in zip(
                mask_preds, det_labels, img_metas):
            if cur_mask_preds.size(0) == 0:
                segm_results.append([[] for _ in range(num_classes)])

            input_h, input_w = img_meta['img_shape'][:2]
            cur_mask_preds = cur_mask_preds[:, :, :input_h, :input_w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                cur_mask_preds = F.interpolate(
                    cur_mask_preds, (ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False)

            cur_mask_preds = cur_mask_preds.squeeze(1) > 0.5
            cur_mask_preds = cur_mask_preds.cpu().numpy().astype(np.uint8)
            cur_labels = cur_labels.detach().cpu().numpy()
            segm_results.append([cur_mask_preds[cur_labels == i]
                                 for i in range(num_classes)])
        return segm_results

    def simple_test_mask(self,
                         mask_feat,
                         det_labels,
                         det_params,
                         det_coors,
                         det_level_inds,
                         det_img_inds,
                         img_metas,
                         num_classes,
                         dynamic_weight,
                         gt_anns_id,
                         others,
                         gt_true_bboxes,
                         gt_masks,
                         rescale=False):
        num_imgs = len(img_metas)
        num_inst_list = [len(l) for l in det_labels]
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        mask_preds = self.forward(mask_feat, det_params, det_coors, det_level_inds,
                                  det_img_inds)

        # mask_preds = mask_preds.sigmoid()
        mask_preds = self.get_mask(mask_preds, others, img_metas, test=True)
        # import cv2
        # cv2.namedWindow("ims1", 0)
        # cv2.namedWindow("gt", 0)
        # cv2.moveWindow("ims1", 2500, 100)
        # cv2.moveWindow("gt", 3200, 100)
        # filename = img_metas[0]['filename']
        # igs = cv2.imread(filename)
        # a = (mask_preds[0, 0]).float()
        # cv2.imshow('ims1', np.array(a.detach().cpu()))
        # cv2.imshow('gt', igs)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()
        # masks_mean_iou, boxes_mean_iou, pseudo_boxes, mask_pred_list = self.mask_mean_iou(mask_preds, gt_masks,
        #                                                                                   gt_true_bboxes, None, None)
        #
        # n = len(torch.cat(gt_true_bboxes))
        # self.iou_mask_sum += n * masks_mean_iou
        # self.iou_box_sum += n * boxes_mean_iou
        # self.iou_num += n
        # # self.iou_mask_sum+=masks_mean_iou
        # # self.iou_box_sum+=boxes_mean_iou
        # # self.iou_num+=1
        # print(self.iou_box_sum / self.iou_num, self.iou_mask_sum / self.iou_num)

        mask_preds = aligned_bilinear(mask_preds, self.out_stride)
        pseudo_bboxes = self.masks_to_boxes(mask_preds)
        pseudo_bboxes_list = torch.split(pseudo_bboxes, num_inst_list, dim=0)

        det_results = []
        segm_results = []
        mask_preds = torch.split(mask_preds, num_inst_list, dim=0)
        # from mmdet.core.point.p2b_utils.crf import dense_crf_2d
        # from PIL import Image
        for cur_mask_preds, cur_labels, img_meta in zip(
                mask_preds, det_labels, img_metas):
            if cur_mask_preds.size(0) == 0:
                segm_results.append([[] for _ in range(num_classes)])

            input_h, input_w = img_meta['img_shape'][:2]
            cur_mask_preds = cur_mask_preds[:, :, :input_h, :input_w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                cur_mask_preds = F.interpolate(
                    cur_mask_preds, (ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False)

            with_crf = False
            if with_crf:
                fn = img_meta['filename']
                img = np.array(Image.open(fn)).astype(np.uint8)
                crf_list = []
                for i in range(len(cur_mask_preds)):
                    crf_pred = dense_crf_2d(img, np.array(cur_mask_preds[i, 0].cpu()))
                    crf_list.append(crf_pred)
                crf_list = np.stack(crf_list)
                cur_mask_preds = torch.tensor(crf_list).to(mask_feat.device) > 0.5

            else:
                cur_mask_preds = cur_mask_preds.squeeze(1) > 0.5
            pseudo_bbox = self.masks_to_boxes(cur_mask_preds)

            det_results.append(pseudo_bbox)
            cur_mask_preds = cur_mask_preds.cpu().numpy().astype(np.uint8)
            cur_labels = cur_labels.detach().cpu().numpy()

            segm_results.append([cur_mask_preds[cur_labels == i]
                                 for i in range(num_classes)])

        # a,b= torch.cat(det_results), torch.cat(gt_true_bboxes)
        # scale_factor = a.new_tensor(scale_factors[0]).unsqueeze(0).repeat(
        #     1,
        #     a.size(-1) // 4)
        # b=b/scale_factor
        #
        # iou_box = bbox_overlaps(a,b, is_aligned=True)
        # n=len(torch.cat(gt_true_bboxes))
        # self.iou_num+=n
        # self.iou_box_sum+=iou_box.sum()
        # print(self.iou_box_sum/self.iou_num)

        det_bboxes, det_labels = self.pseudobox_to_result(det_results, det_labels, dynamic_weight, gt_anns_id,
                                                          scale_factors, rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        num_classes)
            for i in range(len(det_bboxes))
        ]
        return bbox_results, segm_results

    # def simple_test_mask(self, mask_preds, ):
    def pseudobox_to_result(self, pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id, scale_factors, rescale):
        det_bboxes = []
        det_labels = []
        batch_gt = [len(b) for b in gt_labels]
        dynamic_weight = torch.split(dynamic_weight, batch_gt)
        for i in range(len(pseudo_boxes)):
            boxes = pseudo_boxes[i]
            labels = gt_labels[i]

            # if rescale and boxes.shape[0] > 0:
            #     scale_factor = boxes.new_tensor(scale_factors[i]).unsqueeze(0).repeat(
            #         1,
            #         boxes.size(-1) // 4)
            #     boxes /= scale_factor
            #     #gt_box/=scale_factor

            boxes = torch.cat([boxes, dynamic_weight[i].sum(dim=1, keepdim=True)], dim=1)
            gt_anns_id_single = gt_anns_id[i].float()
            boxes = torch.cat([boxes, gt_anns_id_single.unsqueeze(1)], dim=1)
            det_bboxes.append(boxes)
            det_labels.append(labels)
        return det_bboxes, det_labels

    @force_fp32(apply_to=('mask_logits',))  # TODO add the sem_loss
    def loss(self,
             imgs,
             img_metas,
             mask_logits,
             gt_inds,
             gt_bboxes,
             gt_true_bboxes,
             gt_masks,
             gt_labels, dynamic_weight, pseudo_boxes_shake=None):
        # if pseudo_boxes_shake == None:
        self._iter += 1
        similarities, gt_bitmasks, bitmasks_full, bitmasks_weight, \
        gt_bitmasks_shake_list = self.get_targets(gt_bboxes, gt_masks, imgs,
                                                  img_metas,
                                                  dynamic_weight,
                                                  pseudo_boxes_shake)
        if self.with_class_embedding:
            mask_scores = mask_logits.sigmoid()[:, 0, None]
            mask_logits_ = mask_logits[:, 0, None]
            other_mask_logits_ = mask_logits[:, 1, None]
            bg_mask_logits_ = mask_logits[:, 2, None]
        else:
            mask_scores = mask_logits.sigmoid()
            mask_logits_ = mask_logits
        gt_bitmasks = torch.cat(gt_bitmasks, dim=0)
        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)
        bitmasks_weight = torch.cat(bitmasks_weight, dim=0)
        bitmasks_weight = bitmasks_weight[gt_inds].unsqueeze(1).to(mask_scores.dtype)

        losses = {}

        if len(mask_scores) == 0:  # there is no instances detected
            dummy_loss = 0 * mask_scores.sum()
            if not self.boxinst_enabled:
                losses["loss_mask"] = dummy_loss
            else:
                losses["loss_prj"] = dummy_loss
                losses["loss_pairwise"] = dummy_loss

        if self.boxinst_enabled:
            img_color_similarity = torch.cat(similarities, dim=0)
            img_color_similarity = img_color_similarity[gt_inds].to(dtype=mask_scores.dtype)

            loss_prj_term = compute_project_term(mask_scores, gt_bitmasks, dynamic_weight, bitmasks_weight, )

            pairwise_losses = compute_pairwise_term(mask_logits_, self.pairwise_size, self.pairwise_dilation)

            # weights = (img_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
            weights = (img_color_similarity >= self.pairwise_color_thresh).float() * (
                    (1 + gt_bitmasks - bitmasks_weight) > 0).float()
            pairwise_losses = dynamic_weight[:, None, None, None] * pairwise_losses
            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            warmup_factor1 = min(self._iter.item() / float(self._warmup_iters), 1.0)
            warmup_factor2 = min(self._iter.item() / float(self._warmup_iters * 2), 1.0)
            loss_prj_term = loss_prj_term * warmup_factor1
            loss_pairwise = loss_pairwise * warmup_factor2

            losses.update({
                "loss_prj": loss_prj_term,
                "loss_pairwise": loss_pairwise,
            })
            if self.with_class_embedding:
                neg_gt_bitmasks = gt_bitmasks.new_zeros(gt_bitmasks.shape)
                other_mask_losses = self.loss_mask(other_mask_logits_, neg_gt_bitmasks).float().detach()
                bg_mask_losses = self.loss_mask(bg_mask_logits_, neg_gt_bitmasks).float().detach()
                loss_cls_embedding = sum((other_mask_losses + bg_mask_losses) * dynamic_weight) / sum(dynamic_weight)
                losses["loss_cls_embedding"] = loss_cls_embedding * warmup_factor2 * 0.5

            # mask_losses = self.loss_mask(mask_logits_ref, (mask_scores[:, 0] > 0.5).float().detach())
            # if self._iter.item() > self._warmup_iters :
            #     mask_losses = dice_coefficient(mask_scores, (mask_scores[:, 0] > 0.5).float().detach())
            #     loss_mask = sum(mask_losses * dynamic_weight) / sum(dynamic_weight)
            #     losses["loss_mask"] = loss_mask * warmup_factor2
            # else:
            #     mask_losses = dice_coefficient(mask_scores, (mask_scores[:, 0] > 0.5).float().detach())
            #     loss_mask = sum(mask_losses * dynamic_weight) / sum(dynamic_weight)
            #     losses["loss_mask"] = loss_mask * 0

            #
            # if self.mil_proj:
            #     loss_prj_term = compute_project_term(mask_scores_1, gt_bitmasks, dynamic_weight, bitmasks_weight, )
            #
            #     pairwise_losses = compute_pairwise_term(mask_logits_1, self.pairwise_size, self.pairwise_dilation)
            #
            #     loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
            #
            #     loss_prj_term = loss_prj_term * warmup_factor1
            #     loss_pairwise = loss_pairwise * warmup_factor2
            #     losses.update({
            #         "loss_prj_whole": loss_prj_term,
            #         "loss_pairwise_whole": loss_pairwise,
            #     })
            #
            #     # mask_losses = self.loss_mask(mask_logits_ref, (mask_scores[:, 0] > 0.5).float().detach())
            #     mask_losses = dice_coefficient(mask_logits_ref.sigmoid(), (mask_scores[:, 0] > 0.5).float().detach())
            #     loss_mask = sum(mask_losses * dynamic_weight) / sum(dynamic_weight)
            #     losses["loss_mask"] = loss_mask * warmup_factor2
            #     if self.whole_map:
            #         mask_wholemap_losses = dice_coefficient(mask_logits_ref.sigmoid(),
            #                                                 (mask_wholemap_scores > 0.5).float().detach())
            #         loss_mask_wholemap = sum(mask_wholemap_losses * dynamic_weight) / sum(dynamic_weight)
            #         losses["loss_mask_wholemap"] = loss_mask_wholemap * warmup_factor2 * 0.5

        else:
            mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
            loss_mask = mask_losses.mean()
            losses["loss_mask"] = loss_mask

        return losses

    def get_targets(self, gt_bboxes, gt_masks, img, img_metas, dynamic_weight, pseudo_boxes_shake=None):
        """get targets"""

        if self.boxinst_enabled:

            padded_image_masks = []
            padded_images = []

            for i in range(len(img_metas)):
                original_image_masks = torch.ones(img_metas[i]['img_shape'][:2], dtype=torch.float32, device=img.device)

                im_h = img_metas[i]['ori_shape'][0]
                pixels_removed = int(
                    self.bottom_pixels_removed * float(img_metas[i]['img_shape'][0]) / float(im_h)
                )
                if pixels_removed > 0:
                    original_image_masks[-pixels_removed:, :] = 0

                padding = (0, img.shape[3] - img_metas[i]['img_shape'][1],
                           0, img.shape[2] - img_metas[i]['img_shape'][0])  # [left, right, top, bottom]

                padded_image_mask = F.pad(original_image_masks, pad=padding)
                padded_image_masks.append(padded_image_mask)

                original_image = get_original_image(img[i], img_metas[i])  # get RGB image tensor
                # original_image = original_image.to(img.device)
                padded_image = F.pad(original_image, pad=padding)
                padded_images.append(padded_image)

            padded_image_masks = torch.stack(padded_image_masks, dim=0)
            padded_images = torch.stack(padded_images, dim=0)

            similarities, bitmasks, bitmasks_full, bitmasks_weight = self.get_bitmasks_from_boxes(gt_bboxes,
                                                                                                  padded_images,
                                                                                                  padded_image_masks,
                                                                                                  dynamic_weight,
                                                                                                  with_similarity=True)
            if pseudo_boxes_shake:
                bitmasks_shake_list = []

                for shake_box in pseudo_boxes_shake:
                    _, bitmasks_shake, _, _ = self.get_bitmasks_from_boxes(shake_box,
                                                                           padded_images,
                                                                           padded_image_masks,
                                                                           dynamic_weight,
                                                                           with_similarity=False)
                    bitmasks_shake_list.append(bitmasks_shake)

        else:
            start = int(self.out_stride // 2)
            bitmasks = []
            for mask in gt_masks:
                bitmasks.append(mask[:, start::self.out_stride, start::self.out_stride])

            similarities = None
            bitmasks_full = gt_masks
            bitmasks_weight = None
        if pseudo_boxes_shake:
            return similarities, bitmasks, bitmasks_full, bitmasks_weight, bitmasks_shake_list
        else:
            return similarities, bitmasks, bitmasks_full, bitmasks_weight, None

    def get_bitmasks_from_boxes(self, gt_bboxes, padded_images, padded_image_masks, dynamic_weight,
                                with_similarity=True):
        h, w = padded_images.shape[2:]
        stride = self.out_stride
        start = int(stride // 2)

        assert padded_images.size(2) % stride == 0
        assert padded_images.size(3) % stride == 0

        if with_similarity:
            downsampled_images = F.avg_pool2d(padded_images.float(), kernel_size=stride, stride=stride, padding=0)
            downsampled_image_masks = padded_image_masks[:, start::stride, start::stride]
            similarities = []
        else:
            similarities = None

        bitmasks = []
        bitmasks_full = []
        bitmasks_weight = []
        dynamic_weight = dynamic_weight.split([len(i) for i in gt_bboxes])
        for i, per_img_gt_bboxes in enumerate(gt_bboxes):
            if with_similarity:
                image_lab = color.rgb2lab(downsampled_images[i].byte().permute(1, 2, 0).cpu().numpy())
                image_lab = torch.as_tensor(image_lab, device=padded_image_masks.device, dtype=torch.float32)
                image_lab = image_lab.permute(2, 0, 1)[None]
                image_color_similarity = get_image_color_similarity(
                    image_lab, downsampled_image_masks[i],
                    self.pairwise_size, self.pairwise_dilation
                )
                similarities.append(torch.cat([image_color_similarity for _ in range(len(per_img_gt_bboxes))], dim=0))
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            per_im_bitmasks_weight = []

            for j, per_box in enumerate(per_img_gt_bboxes):  # [x1,y1, x2, y2]
                bitmask_full = torch.zeros((h, w), device=per_box.device).float()
                bitmask_full[int(per_box[1]): int(per_box[3]) + 1, int(per_box[0]):int(per_box[2]) + 1] = 1.0
                bitmask_full_weight = torch.zeros((h, w), device=per_box.device).float()

                rate = 0.1
                # bitmask_full_weight[int(per_box[1]): int(per_box[3]) + 1, int(per_box[0]):int(per_box[2]) + 1] = 1.0
                bw, bh = per_box[2] - per_box[0], per_box[3] - per_box[1]
                bitmask_full_weight[
                max(0, int(per_box[1] + rate * bh)): min(int(torch.ceil(per_box[3] - rate * bh) + 1), h),
                max(0, int(per_box[0] + rate * bw)): min(int(torch.ceil(per_box[2] - rate * bw) + 1), w)] += 1.0

                bitmask_full_weight[
                max(0, int(per_box[1] - rate * bh)): min(int(torch.ceil(per_box[3] + rate * bh) + 1), h),
                max(0, int(per_box[0] - rate * bw)): min(int(torch.ceil(per_box[2] + rate * bw) + 1), w)] += 1.0
                # bitmask_full_weight[
                # max(0, int(per_box[1])): min(int(torch.ceil(per_box[3]) + 1), h),
                # max(0, int(per_box[0])): min(int(torch.ceil(per_box[2]) + 1), w)] += 1.0
                bitmask_full_weight = bitmask_full_weight != 1.0
                bitmask_weight = bitmask_full_weight[start::stride, start::stride]

                if self.soft_project:
                    weight = dynamic_weight[i][j]
                    b_w, b_h = per_box[2] - per_box[0], per_box[3] - per_box[1]
                    w_k_p = kernel_and_pad(b_w / 2 * (1 - weight))
                    h_k_p = kernel_and_pad(b_h / 2 * (1 - weight))
                    bitmask_full = \
                        F.avg_pool2d(bitmask_full[None, ...], (w_k_p[0], h_k_p[0]), 1, padding=(w_k_p[1], h_k_p[1]))[0]

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == h
                assert bitmask.size(1) * stride == w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)
                per_im_bitmasks_weight.append(bitmask_weight)

            per_im_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

            bitmasks.append(per_im_bitmasks)
            bitmasks_full.append(per_im_bitmasks_full)
            if with_similarity:
                per_im_bitmasks_weight = torch.stack(per_im_bitmasks_weight, dim=0)
                bitmasks_weight.append(per_im_bitmasks_weight)

        return similarities, bitmasks, bitmasks_full, bitmasks_weight

    def refiner_forward(self, mask_pred, ori_feat, params, img_metas, img_inds):

        mask_feat = ori_feat[img_inds]
        params2 = [i[:, self.num_gen_params:] for i in params]
        weights2, biases2 = self.parse_dynamic_params(torch.cat(params2))
        N, _, H, W = mask_pred.shape
        # mask_pred = mask_pred.reshape(N, 1, H, W)
        mask_feat = aligned_bilinear(mask_feat, int(self.in_stride / self.out_stride))
        x2 = torch.cat([mask_pred, mask_feat], dim=1).reshape(1, -1, H, W)
        for i, (w, b) in enumerate(zip(weights2, biases2)):
            if i == 0:
                w = w[:, 1:]
            x2 = F.conv2d(x2, w, bias=b, stride=1, padding=0, groups=N)
            if i < self.dynamic_convs - 1:
                x2 = F.relu(x2)
        return x2.permute(1, 0, 2, 3)

    def refiner_loss(self, mask_logics, gt_masks, label_weights):
        losses = {}
        mask_preds = mask_logics.sigmoid()
        mask_losses = dice_coefficient(mask_preds, gt_masks.float().detach())
        loss_mask = sum(mask_losses * label_weights) / sum(label_weights)
        warmup_factor = min(self._iter.item() / float(self._warmup_iters * 2), 1.0)
        losses["loss_refiner_mask"] = loss_mask * warmup_factor
        return losses
