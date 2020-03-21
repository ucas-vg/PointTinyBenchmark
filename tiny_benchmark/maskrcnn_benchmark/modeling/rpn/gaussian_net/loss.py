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
from maskrcnn_benchmark.layers.sigmoid_focal_loss import FixSigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


INF = 100000000


class TargetGenerator(object):
    def __init__(self, beta, num_classes, object_sizes_of_interest, label_radius=1.0):
        self.beta = beta
        self.inflection_point = 0.25
        self.num_classes = num_classes
        beta = self.beta
        self.sigma = self.inflection_point * ((beta / (beta - 1)) ** (1.0/beta))
        self.object_sizes_of_interest = object_sizes_of_interest
        self.eps = 1e-6
        self.label_radius = label_radius

    def __call__(self, locations, targets):
        object_sizes_of_interest = self.object_sizes_of_interest
        cls_labels = []
        matched_gt_idxs = []
        self.care = [0, 0]
        for l, locations_level in enumerate(locations):
            cls_label, matched_gt_idx = self.prepare_target_per_level(locations_level, targets, object_sizes_of_interest[l], l)
            cls_label = torch.cat(cls_label, dim=0)   # cat all image label together
            matched_gt_idx = torch.cat(matched_gt_idx, dim=0)
            cls_labels.append(cls_label)
            matched_gt_idxs.append(matched_gt_idx)
        return cls_labels, matched_gt_idxs

    def prepare_target_per_level(self, locations, targets, object_sizes, level=0):
        """
            match_gt_idx = match_gt_idxs[img_idx]
            match_gt_idx[loc_idx, class_id-1] = {  # class_id start from 1
                -1, if no object match
                object_idx, if match object with object_idx
            }
        """
        beta = self.beta
        sigma = self.sigma

        cls_labels = []
        matched_gt_idxs = []
        xs, ys = locations[:, 0], locations[:, 1]
        fpn_stride = xs[1] - xs[0]
        for im_i in range(len(targets)):
            # select object for this fpn level
            targets_per_im = targets[im_i]
            targets_per_im = targets_per_im.convert('xywh')
            sizes = torch.sqrt((targets_per_im.bbox[:, 2] * targets_per_im.bbox[:, 3]))

            min_e = targets_per_im.bbox[:, [2, 3]].min(dim=1)[0] / fpn_stride
            # cause cross method need 3 point with same x or y.
            is_card1_in_the_level = (sizes <= object_sizes[1]) & (sizes > object_sizes[0])
            # is_card2_in_the_level = min_e < 6
            # if level > 0:
            #     is_card1_in_the_level = is_card1_in_the_level & (min_e >= 3)
            #     is_card2_in_the_level = (min_e >= 3) & is_card2_in_the_level
            # is_card_in_the_level = is_card1_in_the_level | is_card2_in_the_level
            is_card_in_the_level = is_card1_in_the_level
            targets_per_im = targets_per_im[is_card_in_the_level]

            cls_label = torch.zeros(size=(len(xs), self.num_classes), device=xs.device)
            matched_gt_idx = torch.zeros(size=(len(xs), self.num_classes), device=xs.device).long() - 1
            if len(targets_per_im) == 0:
                cls_labels.append(cls_label)
                matched_gt_idxs.append(matched_gt_idx)
                continue

            # get gt-boxes infos
            targets_per_im = targets_per_im.convert('xyxy')
            bboxes = targets_per_im.bbox
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            W = bboxes[:, 2] - bboxes[:, 0] + 1
            H = bboxes[:, 3] - bboxes[:, 1] + 1

            # match locations to bbox one by one, and get the score
            D = ((xs[:, None] - cx[None, :]).abs() / (sigma * W[None, :])) ** beta + \
                ((ys[:, None] - cy[None, :]).abs() / (sigma * H[None, :])) ** beta
            Q = torch.exp(-D)

            # clip gaussian range: in boxes make it positive, or get negative
            Q = Q * self.is_in_boxes(xs, ys, bboxes)

            # generate label map
            dis = (xs[:, None] - cx[None, :]) ** 2 + (ys[:, None] - cy[None, :]) ** 2
            labels_per_im = targets_per_im.get_field("labels").to(xs.device)
            card_idx = torch.nonzero(is_card_in_the_level).squeeze(dim=1)
            for c in set(labels_per_im):
                targets_the_class = torch.nonzero(labels_per_im == c).squeeze(dim=1)
                Qc = Q[:, targets_the_class]
                cls_label[:, c-1], m_idx = Qc.max(dim=1)
                matched_gt_idx[:, c-1] = torch.where(
                    cls_label[:, c-1] > self.eps, card_idx[targets_the_class[m_idx]], torch.LongTensor([-1]).to(m_idx.device))
            matched_gt_idxs.append(matched_gt_idx)
            cls_labels.append(cls_label)
        return cls_labels, matched_gt_idxs

    def is_in_boxes(self, xs, ys, bboxes):
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        W = bboxes[:, 2] - bboxes[:, 0] + 1
        H = bboxes[:, 3] - bboxes[:, 1] + 1

        W = (W * self.label_radius).clamp(3)
        H = (H * self.label_radius).clamp(3)

        x1 = cx - (W - 1) / 2
        x2 = cx + (W - 1) / 2
        y1 = cy - (H - 1) / 2
        y2 = cy + (H - 1) / 2

        l = xs[:, None] - x1[None]
        t = ys[:, None] - y1[None]
        r = x2[None] - xs[:, None]
        b = y2[None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
        is_in_boxes = (reg_targets_per_im.min(dim=2)[0] > 0).float()
        return is_in_boxes


class GAULossComputation(object):
    """
    This class computes the Gaussian Net losses.
    """

    def __init__(self, cfg):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]

        self.prepare_targets = TargetGenerator(cfg.MODEL.GAU.LABEL_BETA,
                                               cfg.MODEL.GAU.NUM_CLASSES-1,
                                               object_sizes_of_interest,
                                               cfg.MODEL.GAU.LABEL_RADIUS)
        self.cls_loss_func = FixSigmoidFocalLoss(
            cfg.MODEL.GAU.LOSS_GAMMA,
            cfg.MODEL.GAU.LOSS_ALPHA,
            self.prepare_targets.sigma,
            cfg.MODEL.GAU.FPN_STRIDES,
            cfg.MODEL.GAU.C
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss()
        # self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.vis_labels = cfg.MODEL.GAU.DEBUG.VIS_LABELS

    def valid_pos(self, matches):
        """
        :param match: list[Tensor], each Tensor shape is (B, H, W, C), list len is len(fpn_levels)
        :return:
        """
        valids = []
        for l, match in enumerate(matches):
            center = match[:, 1:-1, 1:-1, :]
            top = match[:, :-2, 1:-1, :]
            bottom = match[:, 2:, 1:-1, :]
            left = match[:, 1:-1, :-2, :]
            right = match[:, 1:-1, 2:, :]
            valid = (left == right) & (top == bottom) & (left == top) & (left == center) & (center >= 0)
            valids.append(valid)
        return valids

    def reshape(self, labels, cls_logits, gau_logits, matched_gt_idxs):
        """
        list of flatten tensor shape (B, M) to list of shape(B, H, W, C)
        :param labels:
        :param cls_logits:
        :param matched_gt_idxs:
        :return:
        """
        num_classes = cls_logits[0].size(1)
        cls_flatten = []
        gau_flatten = []
        labels_flatten = []
        matched_flatten = []
        for l in range(len(labels)):
            N, C, H, W = cls_logits[l].shape
            cls_flatten.append(cls_logits[l].permute(0, 2, 3, 1))
            gau_flatten.append(gau_logits[l].permute(0, 2, 3, 1))
            labels_flatten.append(labels[l].reshape(N, H, W, num_classes))
            matched_flatten.append(matched_gt_idxs[l].reshape(N, H, W, num_classes))
        return cls_flatten, gau_flatten, labels_flatten, matched_flatten

    def __call__(self, locations, logits, targets):
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
        cls_logits, gau_logits = logits
        N = cls_logits[0].size(0)
        labels, matched_gt_idxs = self.prepare_targets(locations, targets)

        # list of flatten tensor shape (B, M) to list of shape(B, H, W, C)
        cls_logits, gau_logits, lables, matched_gt_idxs = self.reshape(labels, cls_logits, gau_logits, matched_gt_idxs)
        valids_pos = self.valid_pos(matched_gt_idxs)

        show_label_map(lables, matched_gt_idxs, valids_pos, cls_logits, gau_logits)
        if self.vis_labels: show_label_map(lables, matched_gt_idxs, valids_pos, cls_logits, gau_logits)

        # box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        # labels_flatten = torch.cat(labels_flatten, dim=0)
        #
        # pos_inds = torch.nonzero(labels_flatten > 0)
        # norm = labels_flatten[pos_inds].sum() + N
        # neg_loss, pos_loss = self.cls_loss_func(
        #     box_cls_flatten,
        #     labels_flatten
        # )
        # neg_loss /= norm / 200
        # pos_loss /= norm  # add N to avoid dividing by a zero

        loss = {}
        # norm = sum([label.sum() / (4**i) for i, label in enumerate(labels_flatten)]) + N
        norm = sum([label.sum() for i, label in enumerate(lables)]) + N
        npos = sum([(label > 0).sum() for i, label in enumerate(lables)]) + N
        losses_fpn = [0.] * 4  # 6
        norms = [npos, npos] + [norm] * 2  # 4
        if self.vis_labels:
            print(sum([(label > 0).sum() for i, label in enumerate(lables)]), '+', end='')
            print(sum([label.sum() for i, label in enumerate(lables)]), '+', end='')
            print(norm, )
        for i, (label, box_cls, gau) in enumerate(zip(lables, cls_logits, gau_logits)):
            losses = self.cls_loss_func(box_cls, gau, label, valids_pos[i])
            for i in range(len(losses)):
                losses_fpn[i] += losses[i] / norms[i]
            # loss.update({
            #     "neg_loss{}".format(i): neg_loss / norm,
            #     "pos_loss{}".format(i): pos_loss / norm
            # })
        loss.update({
            "neg_loss": losses_fpn[0],
            "pos_loss": losses_fpn[1],
            "iou_loss": losses_fpn[2],
            "l1_loss": losses_fpn[3],
            # "gau_neg_loss": losses_fpn[2],
            # "gau_loss": losses_fpn[3],
            # "wh_loss": losses_fpn[4],
            # "xy_loss": losses_fpn[5],
        })
        return loss  # neg_losses, pos_losses


def make_gau_loss_evaluator(cfg):
    loss_evaluator = GAULossComputation(cfg)
    return loss_evaluator

iter = 0
def show_label_map(labels, matched_gt_idxs, valids_pos, box_cls, gaus):
    import matplotlib.pyplot as plt
    import numpy as np
    global iter
    iter += 1
    if iter % 20 != 0: return
    new_labels = []
    new_clses = []
    new_match = []
    new_valid = []
    new_gaus = []
    for i, (label, cls) in enumerate(zip(labels, box_cls)):
        new_labels.append(label.cpu().detach().numpy())
        new_clses.append(cls.cpu().detach().numpy())
        new_gaus.append(gaus[i].cpu().detach().numpy())
        new_match.append(matched_gt_idxs[i].float().cpu().detach().numpy())
        new_valid.append(valids_pos[i].float().cpu().detach().numpy())
    labels = new_labels
    box_cls = new_clses
    match = new_match
    valid = new_valid
    gaus = new_gaus

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    N = sum([(label[0].sum(axis=(0, 1)) > 0).sum() for label in labels])
    C = 4
    n = 1
    # print(N)
    plt.figure(figsize=(12, N*4))
    for l, label in enumerate(labels):
        for c in range(label.shape[-1]):
            if label[0, :, :, c].sum() > 0:
                plt.subplot(N, C, n)
                # print(label.shape)
                plt.imshow(np.log(label[0, :, :, c] + 0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal
                plt_str = ("pos_count:{}; {}, {}".format((label[0, :, :, c] > 0).sum(), l, c))
                plt.title(plt_str)

                plt.subplot(N, C, n + 1)
                pred = sigmoid(box_cls[l][0, :, :, c])
                plt.imshow(np.log(pred+0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal, not show absolute value
                plt.title("p: [{:.4f}, {:.4f}]".format(pred.min(), pred.max()))

                plt.subplot(N, C, n + 2)
                pred = sigmoid(gaus[l][0, :, :, c])
                plt.imshow(np.log(pred+0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal, not show absolute value
                plt.title("g: [{:.4f}, {:.4f}]".format(pred.min(), pred.max()))

                plt.subplot(N, C, n + 3)
                pred = (sigmoid(gaus[l][0, :, :, c]) * sigmoid(box_cls[l][0, :, :, c])) ** 0.5
                plt.imshow(np.log(pred+0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal, not show absolute value
                plt.title("(p*g)^(0.5): [{:.4f}, {:.4f}]".format(pred.min(), pred.max()))
                # print(plt_str)
                n += C
    # plt.show()
    plt.savefig("outputs/pascal/gau/tmp/iter_png/{}.png".format(iter))

    # label_map = 0
    # shape, pos_count = None, []
    # label_maps = []
    # for i in range(0, len(labels)):
    #     label_sum = labels[i].sum()
    #     if shape is None:
    #         if label_sum > 0:
    #             shape = labels[i].shape
    #             label = labels[i]
    #             label_maps.append(label)
    #     else:
    #         label = F.upsample(labels[i], shape[2:], mode='bilinear')
    #         # label_map += label
    #         label_maps.append(label)
    #     pos_count.append(int(label_sum.cpu().numpy()))
    # # print(label_map.shape)
    #
    # for i, label_map in enumerate(label_maps):
    #     label_map = label_map[0].permute((1, 2, 0)).cpu().numpy()[:, :, 0].astype(np.float32)
    #     max_l = label_map.max()
    #     if max_l > 0:
    #         label_map /= max_l
    #
    #     plt.subplot(len(label_maps), 1, i+1)
    #     plt.imshow(label_map)
    #     plt.title("pos_count:{} ".format(pos_count))
    # plt.show()
