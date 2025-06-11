import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.point.roi_heads.bbox_heads.MIL_bbox_head_EP2B import Shared2FCInstanceMILHeadEP2B
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import torch.nn.functional as F
from mmdet.models.losses import accuracy
from mmdet.core.point.dcdet_utils.db import DropBlock2D


@HEADS.register_module()
class Shared2FCInstanceMILHeadEPLUS(Shared2FCInstanceMILHeadEP2B):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=[dict(
                         type='Normal',
                         name='fc_cls',
                         std=0.01,
                         bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='fc_retrain_cls',
                             std=0.01,
                             bias_prob=0.01),
                         # dict(
                         #     type='Normal',
                         #     name='fc_retrain_ins',
                         #     std=0.01,
                         #     bias_prob=0.01)
                     ]
                 ),
                 loss_re_cls=dict(type='FocalLoss', use_sigmoid=True,
                                  gamma=2.0, alpha=0.25, loss_weight=1.0),
                 with_retrain_cls=True,
                 with_retrain_2fcs=True,
                 with_retrain_ins=True,
                 *args,
                 **kwargs):
        super(Shared2FCInstanceMILHeadEPLUS, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.with_retrain_2fcs = with_retrain_2fcs
        self.with_retrain_cls = with_retrain_cls
        if with_retrain_2fcs:
            _, self.retrained_fcs, last_layer_dim = \
                self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)

        if with_retrain_cls:
            self.fc_retrain_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.num_classes)
            self.loss_re_cls = build_loss(loss_re_cls)

        if with_retrain_cls:
            self.fc_retrain_ins = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=self.num_classes)
            # self.loss_re_cls = build_loss(loss_re_cls)

        self.fc_distill_reg = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=4)
        self.loss_bbox_ori = build_loss(loss_bbox_ori)

        self.DropBlock2D = DropBlock2D()

    def forward_distill_2fcs(self, x, others=None):
        # x=self.DropBlock2D(x)
        if self.with_retrain_2fcs:
            x = x.flatten(1)
            for fc in self.retrained_fcs:
                x = self.relu(fc(x))
        else:
            if self.num_shared_fcs > 0:
                x = x.flatten(1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
        return x

    def forward_distill_reg(self, x, others=None):
        bbox_pred = self.fc_distill_reg(x)
        return bbox_pred

    def loss_distill(self, bboxes_pred, gt_bboxes, label_weights):
        losses = {}
        # gt_bboxes  = gt_bboxes.unsqueeze(1).expand(reg_bboxes.shape)
        num_gt = len(label_weights)
        num_instance = int(len(bboxes_pred) / num_gt)
        label_weights = label_weights.unsqueeze(-1).expand([num_gt, num_instance]).reshape(-1, 1)
        # loss = self.loss_bbox()
        num_reg_pos = bboxes_pred.shape[0]
        losses['loss_bbox'] = self.loss_bbox_ori(
            bboxes_pred,
            gt_bboxes,
            label_weights,
            avg_factor=num_reg_pos
        )
        return losses

    def forward_retrain_cls(self, x, others=None):
        cls_score = self.fc_retrain_cls(x)
        return cls_score

    def forward_retrain_ins(self, x, others=None):

        ins_score = self.fc_retrain_ins(x)
        return ins_score

    def forward_retrain_neg_ins(self, x, others=None):
        ins_score = self.fc_retrain_neg_ins(x)
        return ins_score

    def loss_retrain_cls(self, cls_scores, gt_labels, label_weights, neg_weights, num_pos, ins_scores=None):
        cls_scores = cls_scores.sigmoid()
        num_gt = len(gt_labels)
        pos_scores, neg_scores = cls_scores[:num_pos], cls_scores[num_pos:]
        if ins_scores is not None:
            ins_scores = ins_scores.reshape(num_gt, -1, self.num_classes)
            ins_scores = ins_scores.softmax(dim=-2)
            pos_scores = pos_scores.reshape(num_gt, -1, self.num_classes)
            pos_scores = (pos_scores * ins_scores).sum(dim=-2)
        else:
            pos_scores = pos_scores.reshape(num_gt, -1, self.num_classes).mean(dim=1)
        cls_scores_ = torch.cat([pos_scores, neg_scores])
        neg_labels = gt_labels.new_full((len(cls_scores) - num_pos,), self.num_classes)
        labels_ = torch.cat([gt_labels, neg_labels])
        neg_weights = neg_weights * label_weights.mean()
        label_weights_ = torch.cat([label_weights, neg_weights])
        # label_weights_ = label_weights.new_full((len(labels_),), label_weights.mean())
        # label_weights_[:num_gt] = label_weights
        loss_cls_ = self.loss_re_cls(
            cls_scores_,
            labels_,
            label_weights_,
            avg_factor=num_gt)
        acc = accuracy(cls_scores_[:num_gt], labels_[:num_gt])

        return loss_cls_, acc

    def loss_retrain_cls_2(self, cls_scores, gt_labels, label_weights, neg_weights, num_pos, ins_scores=None,
                           neg_ins_scores=None):
        cls_scores = cls_scores.sigmoid()
        num_gt = len(gt_labels)
        pos_scores, neg_scores = cls_scores[:num_pos], cls_scores[num_pos:]
        if ins_scores is not None:
            ins_scores = ins_scores.reshape(num_gt, -1, self.num_classes)
            ins_scores = ins_scores.softmax(dim=-2)
            pos_scores = pos_scores.reshape(num_gt, -1, self.num_classes)
            pos_scores_1 = (pos_scores * ins_scores).sum(dim=-2)
            neg_ins_scores = neg_ins_scores.reshape(num_gt, -1, self.num_classes)
            neg_ins_scores = neg_ins_scores.softmax(dim=-2)
            pos_scores_2 = (pos_scores * neg_ins_scores).sum(dim=-2)
        else:
            pos_scores = pos_scores.reshape(num_gt, -1, self.num_classes).mean(dim=1)
        cls_scores_ = torch.cat([pos_scores_1, pos_scores_2, neg_scores])
        neg_labels = gt_labels.new_full((len(cls_scores_) - num_gt,), self.num_classes)
        labels_ = torch.cat([gt_labels, neg_labels])
        neg_weights = torch.cat([neg_weights, neg_weights.new_full((num_gt,), 1)])
        neg_weights = neg_weights * label_weights.mean()
        label_weights_ = torch.cat([label_weights, neg_weights])
        # label_weights_ = label_weights.new_full((len(labels_),), label_weights.mean())
        # label_weights_[:num_gt] = label_weights
        loss_cls_ = self.loss_re_cls(
            cls_scores_,
            labels_,
            label_weights_,
            avg_factor=num_gt)

        acc = accuracy(cls_scores_[:num_gt], labels_[:num_gt])

        return loss_cls_, acc

    def loss_retrain_cls_3(self, cls_scores, ins_scores, gt_labels, label_weights):

        # cls_scores = cls_scores.softmax(dim=-1)
        # num_gt = len(gt_labels)
        # if ins_scores is not None:
        #     ins_scores = ins_scores.reshape(num_gt, -1, self.num_classes)
        #     ins_scores = ins_scores.softmax(dim=-2)
        #     cls_scores = cls_scores.reshape(num_gt, -1, self.num_classes)
        #     cls_scores = (cls_scores * ins_scores).sum(dim=-2)
        # else:
        #     cls_scores = cls_scores.reshape(num_gt, -1, self.num_classes).mean(dim=1)
        # cls_scores_ = cls_scores
        # labels_ = gt_labels
        # label_weights_ = label_weights
        # label_weights_ = label_weights.new_full((len(labels_),), label_weights.mean())
        # label_weights_[:num_gt] = label_weights
        loss_cls_, bag_acc, num_pos = self.loss_re_cls(
            cls_scores.softmax(dim=-1),
            ins_scores,
            gt_labels,
            cls_scores.new_full([*cls_scores.shape[:2]], 1)[:, :, None],
            label_weights[:, None])
        return loss_cls_, bag_acc
    # if isinstance(loss_cls_, dict):
    #     losses.update(loss_cls_)
    # else:
    #     losses['loss_cls'] = loss_cls_
    # if self.custom_activation:
    #     acc_ = self.loss_cls.get_accuracy(cls_score, labels)
    #     losses.update(acc_)
    # else:
    #     losses['acc'] = accuracy(cls_score, labels)
    # pass
