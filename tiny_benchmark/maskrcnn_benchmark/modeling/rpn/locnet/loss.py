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
from maskrcnn_benchmark.layers.sigmoid_focal_loss import FixSigmoidFocalLoss, L2LossWithLogit
from maskrcnn_benchmark.layers.ghm_loss import GHMC
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from .target_generator import *


class LOCLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        # self.cls_loss_func = SigmoidFocalLoss(
        #     cfg.MODEL.LOC.LOSS_GAMMA,
        #     cfg.MODEL.LOC.LOSS_ALPHA
        # )
        cls_loss_name = cfg.MODEL.LOC.CLS_LOSS
        self.cls_divide_pos_num = True
        if cls_loss_name == 'fixed_focal_loss':
            self.cls_loss_func = FixSigmoidFocalLoss(
                cfg.MODEL.LOC.LOSS_GAMMA,
                cfg.MODEL.LOC.LOSS_ALPHA
            )
        elif cls_loss_name == 'L2':
            self.cls_loss_func = L2LossWithLogit()
        elif cls_loss_name == 'GHMC':
            self.cls_loss_func = GHMC(bins=cfg.MODEL.LOC.LOSS_GHMC_BINS,
                              alpha=cfg.MODEL.LOC.LOSS_GHMC_ALPHA,
                              momentum=cfg.MODEL.LOC.LOSS_GHMC_MOMENTUM)
            self.cls_divide_pos_num = False

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        if cfg.MODEL.LOC.TARGET_GENERATOR == 'fcos' and cfg.MODEL.LOC.FCOS_CENTERNESS:
            self.centerness_loss_func = nn.BCEWithLogitsLoss()

        self.prepare_targets = build_target_generator(cfg)
        self.cls_loss_weight = cfg.MODEL.LOC.CLS_WEIGHT
        self.centerness_weight_reg = cfg.MODEL.LOC.TARGET_GENERATOR == 'fcos' and cfg.MODEL.LOC.FCOS_CENTERNESS_WEIGHT_REG
        self.debug_vis_labels = cfg.MODEL.LOC.DEBUG.VIS_LABELS

        self.cls_divide_pos_sum = cfg.MODEL.LOC.DIVIDE_POS_SUM
        if self.cls_divide_pos_sum:
            self.cls_divide_pos_num = False

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
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
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        if self.debug_vis_labels: show_label_map(labels, box_cls)

        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1, num_classes))  # changed
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # class loss
        label_flatten_max = labels_flatten.max(dim=1)[0]
        pos_inds = torch.nonzero(label_flatten_max > 0).squeeze(1)
        pos_sum = labels_flatten.sum()
        cls_losses = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten
        )
        if isinstance(cls_losses, (list,)):
            for i in range(len(cls_losses)):
                if self.cls_divide_pos_num:
                    cls_losses[i] /= (pos_inds.numel() + N)  # add N to avoid dividing by a zero
                elif self.cls_divide_pos_sum:
                    cls_losses[i] /= (pos_sum + N)
        else:
            if self.cls_divide_pos_num:
                cls_losses /= (pos_inds.numel() + N)  # add N to avoid dividing by a zero
            elif self.cls_divide_pos_sum:
                cls_losses /= (pos_sum + N)

        # reg loss
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]

        if pos_inds.numel() > 0:
            if self.centerness_weight_reg:
                reg_weights = centerness_targets = self.prepare_targets.compute_centerness_targets(reg_targets_flatten)
            else:
                reg_weights = label_flatten_max[pos_inds]
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                reg_weights
            )
        else:
            reg_loss = box_regression_flatten.sum()

        if isinstance(cls_losses, (list,)):
            losses = {"loss_cls{}".format(i): cls_loss * self.cls_loss_weight for i, cls_loss in enumerate(cls_losses)}
            losses['loss_reg'] = reg_loss
        else:
            losses = {
                "loss_cls": cls_losses * self.cls_loss_weight,
                "loss_reg": reg_loss
            }

        # centerness loss
        if centerness is not None:
            centerness_flatten = [centerness[l].reshape(-1) for l in range(len(centerness))]
            centerness_flatten = torch.cat(centerness_flatten, dim=0)
            centerness_flatten = centerness_flatten[pos_inds]

            if pos_inds.numel() > 0:
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_targets
                )
            else:
                centerness_loss = centerness_flatten.sum()

            losses["loss_centerness"] = centerness_loss
        return losses


def make_location_loss_evaluator(cfg):
    loss_evaluator = LOCLossComputation(cfg)
    return loss_evaluator


class LabelMapShower(object):
    def __init__(self, area_ths=1, show_iter=1):
        self.area_ths = area_ths
        self.show_iter = show_iter
        self.counter = 0
        self.merge_levels = True
        self.show_classes = None  # torch.Tensor([15]).long() - 1
        self.merge_method = 'max'  # merge class and fpn levels' method
        assert self.merge_method in ['max', 'sum']

    def __call__(self, labels, box_cls):
        if (self.counter // self.area_ths + 1) % self.show_iter != 0:
            self.counter += 1
            return
        self.counter += 1

        labels = labels.copy()
        for i, (label, cls) in enumerate(zip(labels, box_cls)):
            # labels[i] = (label > 0).float().reshape((2, 1, cls.shape[-2], cls.shape[-1]))
            if self.show_classes is not None:
                label = label[:, self.show_classes]
            if self.merge_method == 'sum':
                labels[i] = label.sum(dim=1)
            elif self.merge_method == 'max':
                labels[i] = label.max(dim=1)[0]
            labels[i] = labels[i].reshape((cls.shape[0], 1, cls.shape[-2], cls.shape[-1]))
        if self.merge_levels:
            label_map = 0
        else:
            label_maps = []
        shape, pos_count = None, []
        for i in range(0, len(labels)):
            label_sum = labels[i].sum()
            if shape is None:
                if label_sum > 0:
                    shape = labels[i].shape
                    label = labels[i]
                    if not self.merge_levels:
                        label_maps.append(label)
                    else:
                        label_map = label
            elif label_sum > 0:
                if self.merge_levels:
                    label = F.upsample(labels[i], shape[2:], mode='bilinear')
                    if self.merge_method == 'max':
                        label_map = torch.max(torch.stack([label_map, label]), dim=0)[0]
                    elif self.merge_method == 'sum':
                        label_map += label
                else:
                    label_maps.append(labels[i])
            pos_count.append(int(label_sum.cpu().numpy()))
        # print(label_map.shape)
        import matplotlib.pyplot as plt
        import numpy as np
        if self.merge_levels:
            label_maps = [label_map]
        else:
            # ms = max([max(label_map.shape) for label_map in label_maps])
            plt.figure(figsize=(5*len(label_maps), 5*1))
        for i, label_map in enumerate(label_maps):
            label_map = F.upsample(label_map, (140, 100), mode='bilinear')
            label_map = label_map[0].permute((1, 2, 0)).cpu().numpy()[:, :, 0].astype(np.float32) ** 2
            max_l = label_map.max()
            if max_l > 0:
                label_map /= max_l

            if len(label_maps) > 1:
                plt.subplot(1, len(label_maps), i + 1)
            plt.imshow(label_map)
            plt.title("pos_count:{} ".format(pos_count))
        plt.show()


show_label_map = LabelMapShower()
