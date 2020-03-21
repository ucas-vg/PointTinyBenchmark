# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.ohem_loss import OHEMLoss, OHEM2Loss
from maskrcnn_benchmark.structures.bounding_box import BoxList


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func, ohem_loss=None):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']
        self.ohem_loss = ohem_loss
        # self.debug = False   # add by hui

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        # ################# changed by hui ###################################################
        if len(target.bbox) == 0:
            matched_idxs = torch.LongTensor([-1] * match_quality_matrix.shape[1])
        else:
            matched_idxs = self.proposal_matcher(match_quality_matrix)

        # for anchor recall cal
        # if self.debug:
        #     record_for_recall(matched_idxs, target)
        #####################################################################################

        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        # HxWxSxA
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        # add by hui ###############################################
        # _box_regression = box_regression.reshape((len(targets), -1, box_regression.shape[-1]))
        # for box_regression_per_img, anchors_per_image, targets_per_img in zip(_box_regression, anchors, targets):
        #     assert len(anchors_per_image) == len(box_regression_per_img)
        #     pred_boxes = self.box_coder.decode(box_regression_per_img, anchors_per_image.bbox)
        #     pred_boxes = BoxList(pred_boxes, targets_per_img.size, mode='xyxy')
        #     ious = boxlist_iou(targets_per_img, pred_boxes)
        #     ious
        # #########################################################

        labels, regression_targets = self.prepare_targets(anchors, targets)

        # show_label(anchors[0].size, labels, regression_targets, objectness)

        # sample
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        )
        # raise ValueError("(sampled_inds.numel()) devide twice, another time is in line 156")

        # ################################# add by hui ###################################################
        if self.ohem_loss is None:
            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds]
            )
            box_loss = box_loss / (sampled_inds.numel())
        #             print('rpnx', sampled_inds.numel())
        else:
            objectness_loss = self.ohem_loss(objectness[sampled_inds], labels[sampled_inds])
            box_loss = box_loss / self.ohem_loss.sample_count
        #             print('rpn', self.ohem_loss.sample_count)
        # #################################################################################################

        return objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    # #################################### changed by hui ####################################################
    ohem_loss = None
    if cfg.MODEL.RPN.OHEM == 1:
        ohem_loss = OHEMLoss(cfg.MODEL.RPN.OHEM1_NEG_RATE, binary_logits=True)
    elif cfg.MODEL.RPN.OHEM == 2:
        ohem_loss = OHEM2Loss(cfg.MODEL.RPN.OHEM2_BATCH_SIZE_PER_IM * cfg.SOLVER.IMS_PER_BATCH // cfg.SOLVER.NUM_GPU,
                              cfg.MODEL.RPN.OHEM2_POSITIVE_FRACTION, binary_logits=True,
                              hard_rate=cfg.MODEL.RPN.OHEM2_HARD_RATE)

    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, generate_rpn_labels, ohem_loss)
    # #######################################################################################################

    return loss_evaluator

# ############## add by hui #########################################################3


def record_for_recall(matched_idxs, target):
    matched_proposal = matched_idxs.clone().cpu().numpy().tolist()
    target_length = target.get_field("labels").size()[0]
    # matched_proposal = tuple(set(matched_proposal[:-target_length]))
    matched_proposal = [i for i in matched_proposal if i > 0]

    from MyPackage.tools.debug_log.recorder import recorder

    bbox = target.bbox
    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1
    scale = torch.sqrt(w * h).cpu().numpy()
    recorder.record('anchor_recall', scale.tolist(),
                    target.get_field("labels").cpu().numpy().tolist(),
                    target_length, matched_proposal, len(matched_proposal))


batch_id = 0
def show_label(img_size, labels, reg_targets, objectness):
    import matplotlib.pyplot as plt
    import numpy as np
    W, H = img_size
    S, A = 1, 3
    stride = (4, 8, 16, 32, 64)

    labels = labels[0].reshape((-1, S, A))
    reg_targets = reg_targets[0].reshape((-1, S, A, 4))
    objectness = objectness.reshape((-1, labels.shape[0], S, A))[0]
    new_labels, new_reg_targets, new_objectness = [], [], []
    sidx = 0
    for s in stride:
        w, h = W // s, H // s
        label = labels[sidx: sidx+w*h].reshape(h, w, S, A)
        new_labels.append(label)
        reg_target = reg_targets[sidx:sidx+w*h].reshape(h, w, S, A, 4)
        new_reg_targets.append(reg_target)
        new_objectness.append(objectness[sidx:sidx+w*h].reshape(h, w, S, A))
        sidx += w * h
    assert sidx == len(labels)
    labels = new_labels
    reg_targets = new_reg_targets
    objectness = new_objectness

    i = 0

    # show labels
    N = 0
    for label in labels:
        label = label.cpu().numpy()
        for s in range(S):
            for a in range(A):
                if np.sum(label[:, :, s, a] > 0) == 0:
                    continue
                N += 1

    i = 1
    plt.figure(figsize=(12, N*4))
    for l, label in enumerate(labels):
        label = label.cpu().numpy()
        for s in range(S):
            for a in range(A):
                npos = np.sum(label[:, :, s, a] > 0)
                if npos == 0:
                    continue
                plt.subplot(N, 2, i)
                plt.imshow((label[:, :, s, a] + 1) / 3, vmin=0, vmax=1)
                plt.title("P{}, S:{}, A:{}, npos:{}".format(l, s, a, npos))
                i += 1

                plt.subplot(N, 2, i)
                plt.imshow((objectness[l][:, :, s, a]).sigmoid().detach().cpu().numpy(), vmin=0, vmax=1)
                plt.title("P{}, S:{}, A:{}".format(l, s, a))
                i += 1
    plt.show()

    valid_reg_targets = []
    for label, reg_target in zip(labels, reg_targets):
        reg_target = reg_target[label == 1]
        valid_reg_targets.append(reg_target)
    valid_reg_targets = torch.cat(valid_reg_targets, dim=0)
    global batch_id
    torch.save(valid_reg_targets, 'outputs/tmp/valid_reg_targets{}.pth'.format(batch_id))
    batch_id += 1
    # if len(valid_reg_targets) > 1:
    #     print(valid_reg_targets.shape[0], valid_reg_targets.mean(dim=0), valid_reg_targets.std(dim=0))
    # else:
    #     print(valid_reg_targets)

# ########################################################3#########################################################3
