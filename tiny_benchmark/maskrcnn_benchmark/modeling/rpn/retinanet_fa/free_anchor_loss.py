"""
This file contains specific functions for computing losses on the RetinaNet
file
"""


import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import cat

# from maskrcnn_benchmark.modeling.matcher import Matcher
from .matcher import Matcher  # changed by hui
from maskrcnn_benchmark.structures.boxlist_ops import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        return x.clamp(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output, None, None


clip = Clip.apply


class FreeAnchorLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.box_coder = box_coder
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        self.iou_threshold = cfg.FREEANCHOR.IOU_THRESHOLD
        self.pre_anchor_topk = cfg.FREEANCHOR.PRE_ANCHOR_TOPK
        self.smooth_l1_loss_param = (cfg.FREEANCHOR.BBOX_REG_WEIGHT, cfg.FREEANCHOR.BBOX_REG_BETA)
        self.bbox_threshold = cfg.FREEANCHOR.BBOX_THRESHOLD
        self.focal_loss_alpha = cfg.FREEANCHOR.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.FREEANCHOR.FOCAL_LOSS_GAMMA

        self.positive_bag_loss_func = positive_bag_loss
        self.negative_bag_loss_func = focal_loss

    def __call__(self, anchors, box_cls, box_regression, targets):
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
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        box_cls_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
        ):
            N, A, H, W = box_cls_per_level.shape
            C = self.num_classes
            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        box_cls = cat(box_cls_flattened, dim=1)
        box_regression = cat(box_regression_flattened, dim=1)

        cls_prob = torch.sigmoid(box_cls)
        box_prob = []
        positive_numels = 0
        positive_losses = []
        for img, (anchors_, targets_, cls_prob_, box_regression_) in enumerate(
                zip(anchors, targets, cls_prob, box_regression)
        ):
            labels_ = targets_.get_field("labels") - 1

            with torch.set_grad_enabled(False):
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                box_localization = self.box_coder.decode(box_regression_, anchors_.bbox)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = boxlist_iou(
                    targets_,
                    BoxList(box_localization, anchors_.size, mode='xyxy')
                )

                t1 = self.bbox_threshold
                # t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                t2 = object_box_iou.max(dim=1, keepdim=True)[0].clamp(min=t1 + 1e-12)  # change by hui for new torch version

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                object_box_prob = (
                    (object_box_iou - t1) / (t2 - t1)
                ).clamp(min=0, max=1)

                indices = torch.stack([torch.arange(len(labels_)).type_as(labels_), labels_], dim=0)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [j, c]
                """
                from "start" to "end" implement:
                
                image_box_iou = torch.sparse.max(object_cls_box_prob, dim=0).t()
                
                """
                # start
                # indices = torch.nonzero(torch.sparse.sum(
                #     object_cls_box_prob, dim=0
                # ).to_dense()).t_()
                indices = torch.nonzero(torch.sparse.sum(
                    object_cls_box_prob, dim=(0,)    # use dim=0 will reduce all, it's a bug for pytorch1.0.0
                ).to_dense()).t_()

                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors_.bbox.size(0), self.num_classes).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (labels_.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor([0]).type_as(object_box_prob)
                    ).max(dim=0)[0]
                    # ).max(dim=0).values   # change by hui

                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]), nonzero_box_prob,
                        size=(anchors_.bbox.size(0), self.num_classes)
                    ).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = boxlist_iou(targets_, anchors_)
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2, labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)
            ).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_object_targets = self.box_coder.encode(targets_.bbox.unsqueeze(dim=1), anchors_.bbox[matched])
            retinanet_regression_loss = smooth_l1_loss(
                box_regression_[matched], matched_object_targets, *self.smooth_l1_loss_param
            )
            matched_box_prob = torch.exp(-retinanet_regression_loss)

            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_numels += len(targets_)
            positive_losses.append(self.positive_bag_loss_func(matched_cls_prob * matched_box_prob, dim=1))

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, positive_numels)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        negative_loss = self.negative_bag_loss_func(
            cls_prob * (1 - box_prob), self.focal_loss_gamma
        ) / max(1, positive_numels * self.pre_anchor_topk)

        losses = {
            "loss_retina_positive": positive_loss * self.focal_loss_alpha,
            "loss_retina_negative": negative_loss * (1 - self.focal_loss_alpha),
        }
        return losses


def smooth_l1_loss(pred, target, weight, beta):
    val = target - pred
    abs_val = val.abs()
    smooth_mask = abs_val < beta
    return weight * torch.where(smooth_mask, 0.5 / beta * val ** 2, (abs_val - 0.5 * beta)).sum(dim=-1)


def positive_bag_loss(logits, *args, **kwargs):
    # bag_prob = Mean-max(logits)
    weight = 1 / clip(1 - logits, 1e-12, None)
    weight /= weight.sum(*args, **kwargs).unsqueeze(dim=-1)
    bag_prob = (weight * logits).sum(*args, **kwargs)
    # positive_bag_loss = -log(bag_prob)
    return F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')


def focal_loss(logits, gamma):
    return torch.sum(
        logits ** gamma * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction='none')
    )


def make_free_anchor_loss_evaluator(cfg, box_coder):
    return FreeAnchorLossComputation(cfg, box_coder)

