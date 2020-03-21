"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import warnings
from math import sqrt


INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cascade_area_th = cfg.MODEL.FCOS.CASCADE_AREA_TH
        self.vis_labels = cfg.MODEL.FCOS.DEBUG.VIS_LABELS
        self.no_match_gt_count = {pos_area: 0 for pos_area in self.cascade_area_th}

    def prepare_targets(self, points, targets, pos_area):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, pos_area
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest, pos_area):
        def is_in_pos_boxes(xs, ys, targets_per_im, pos_area, EPS=1e-6):
            bboxes = targets_per_im.bbox
            centers = torch.cat([(bboxes[:, [0]] + bboxes[:, [2]]) / 2, (bboxes[:, [1]] + bboxes[:, [3]]) / 2], dim=1)
            WH = torch.cat([(bboxes[:, [2]] - bboxes[:, [0]] + 1), (bboxes[:, [3]] - bboxes[:, [1]] + 1)], dim=1)
            WH *= sqrt(pos_area)
            # WH[WH < 2 + EPS] = 2 + EPS
            x1y1 = centers - (WH - 1) / 2
            x2y2 = centers + (WH - 1) / 2
            bboxes = torch.cat([x1y1, x2y2], dim=1)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            ltrb = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = ltrb.min(dim=2)[0] > 0

            self.no_match_gt_count[pos_area] += (is_in_boxes.sum(dim=0) < EPS).sum().item()
            if (self.no_match_gt_count[pos_area] + 1) % 100 == 0:
                import warnings
                warnings.warn("when pos_area={}, already have {} ground-truth no matched."
                              .format(pos_area, self.no_match_gt_count[pos_area]))
            return is_in_boxes

        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            is_in_boxes = is_in_pos_boxes(xs, ys, targets_per_im, pos_area)    # add here

            # assert (is_in_boxes != is_in_boxes2).sum().item() == 0, 'not equal {}vs{}, {}vs{}\n {}, {}'.\
            #     format(is_in_boxes.sum(), is_in_boxes2.sum(), is_in_boxes.device, is_in_boxes2.device,
            #            ((the_bboxes - bboxes).abs() > 1e-6).sum(), ((ltrb - reg_targets_per_im).abs() > 1e-6).sum())

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    # def cal_single_label_loss(self, locations, box_cls, box_regression_flatten, centerness_flatten, targets):
    #     N = box_cls[0].size(0)
    #     num_classes = box_cls[0].size(1)
    #     labels, reg_targets = self.prepare_targets(locations, targets)
    #
    #     if self.vis_labels: show_label_map(labels, box_cls)
    #
    #     box_cls_flatten = []
    #     labels_flatten = []
    #     reg_targets_flatten = []
    #     for l in range(len(labels)):
    #         box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
    #         labels_flatten.append(labels[l].reshape(-1))
    #         reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
    #     box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
    #     labels_flatten = torch.cat(labels_flatten, dim=0)
    #     reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
    #
    #     pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
    #     cls_loss = self.cls_loss_func(
    #         box_cls_flatten,
    #         labels_flatten.int()
    #     ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
    #
    #     box_regression_flatten = box_regression_flatten[pos_inds]
    #     reg_targets_flatten = reg_targets_flatten[pos_inds]
    #     centerness_flatten = centerness_flatten[pos_inds]
    #
    #     if pos_inds.numel() > 0:
    #         centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
    #         reg_loss = self.box_reg_loss_func(
    #             box_regression_flatten,
    #             reg_targets_flatten,
    #             centerness_targets
    #         )
    #         centerness_loss = self.centerness_loss_func(
    #             centerness_flatten,
    #             centerness_targets
    #         )
    #     else:
    #         reg_loss = box_regression_flatten.sum()
    #         centerness_loss = centerness_flatten.sum()
    #     return cls_loss, reg_loss, centerness_loss

    def cal_single_label_loss(self, locations, box_cls, box_regression_flatten, centerness_flatten, targets, pos_area,
                              need_reg=True):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets, pos_area)

        if self.vis_labels: show_label_map(labels, box_cls)

        box_cls_flatten = [box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes) for l in range(len(box_cls))]
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        labels_flatten = [labels[l].reshape(-1) for l in range(len(labels))]
        labels_flatten = torch.cat(labels_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        reg_loss, centerness_loss = 0, 0
        if need_reg:
            reg_targets_flatten = [reg_targets[l].reshape(-1, 4) for l in range(len(reg_targets))]
            reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

            box_regression_flatten = box_regression_flatten[pos_inds]
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            if centerness_flatten is not None: centerness_flatten = centerness_flatten[pos_inds]

            if pos_inds.numel() > 0:
                centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets
                )
                if centerness_flatten is not None:
                    centerness_loss = self.centerness_loss_func(
                        centerness_flatten,
                        centerness_targets
                    )
            else:
                reg_loss = box_regression_flatten.sum()
                if centerness_flatten is not None: centerness_loss = centerness_flatten.sum()
                warnings.warn("no positive sample in this batch.")
        return cls_loss, reg_loss, centerness_loss

    def __call__(self, locations, box_cls_set, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # flatten and cat all fpn level results
        box_regression_flatten = []
        centerness_flatten = []
        for l in range(len(box_regression)):
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            if centerness is not None: centerness_flatten.append(centerness[l].reshape(-1))
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        if centerness is not None: centerness_flatten = torch.cat(centerness_flatten, dim=0)
        else: centerness_flatten = None

        all_cls_loss, all_reg_loss, all_centerness_loss = 0, 0, 0
        for area_th in self.cascade_area_th:
            area_rate = int(area_th * 100)
            box_cls = box_cls_set['cls_logits_{}%'.format(area_rate)]
            need_reg = area_rate == 100
            cls_loss, reg_loss, centerness_loss = self.cal_single_label_loss(locations, box_cls, box_regression_flatten,
                                                                             centerness_flatten, targets, area_th, need_reg)
            all_cls_loss += cls_loss
            all_reg_loss += reg_loss
            all_centerness_loss += centerness_loss
        all_cls_loss /= len(self.cascade_area_th)
        return all_cls_loss, all_reg_loss, all_centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator


def show_label_map(labels, box_cls):
    for i, (label, cls) in enumerate(zip(labels, box_cls)):
        labels[i] = (label > 0).float().reshape((2, 1, cls.shape[-2], cls.shape[-1]))
    label_map = 0
    shape, pos_count = None, []
    for i in range(0, len(labels)):
        label_sum = labels[i].sum()
        if shape is None:
            if label_sum > 0:
                shape = labels[i].shape
        else:
            label = F.upsample(labels[i], shape[2:], mode='bilinear')
            label_map += label
        pos_count.append(int(label_sum.cpu().numpy()))
    # print(label_map.shape)
    import matplotlib.pyplot as plt
    import numpy as np
    label_map = label_map[0].permute((1, 2, 0)).cpu().numpy()[:, :, 0].astype(np.float32)
    max_l = label_map.max()
    if max_l > 0:
        label_map /= max_l
    plt.imshow(label_map)
    plt.title("pos_count:{} ".format(pos_count))
    plt.show()
