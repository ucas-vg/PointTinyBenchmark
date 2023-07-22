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
# from mmdet.core.point.dcdet_utils.db import DropBlock2D
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.core.bbox import bbox_overlaps


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
                     override=dict(
                         type='Normal',
                         name='fc_cls',
                         std=0.01,
                         bias_prob=0.01),
                 ),
                 loss_bbox_ori=dict(
                     type='L1Loss', loss_weight=0.25),
                 loss_anti_iou=dict(type='AntiGIoULoss', loss_weight=0.25),
                 loss_objectness=dict(type='FocalLoss', use_sigmoid=True,
                                      gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_iou=dict(type='SmoothL1Loss', loss_weight=5.0),
                 # loss_objectness=dict(type='QualityFocalLoss', use_sigmoid=True,
                 #             beta=2.0, loss_weight=1.0),

                 with_distill_2fcs=False,
                 with_distill=True,
                 with_reg=False,
                 with_retrain_ins=True,
                 *args,
                 **kwargs):
        super(Shared2FCInstanceMILHeadEPLUS, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)

        _, self.objness_2fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)

        self.with_distill_2fcs = with_distill_2fcs
        self.with_distill = with_distill
        self.fc_distill_reg_ori = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=4)
        self.fc_distill_reg_re = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=4)
        self.fc_objness = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=1)
        self.fc_iou = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=1)
        self.loss_bbox_ori = build_loss(loss_bbox_ori)
        self.loss_anti_iou = build_loss(loss_anti_iou)
        self.loss_obj = build_loss(loss_objectness)
        self.loss_iou = build_loss(loss_iou)
        # self.DropBlock2D = DropBlock2D()
        if self.with_distill:
            _, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(0, self.num_reg_fcs, self.shared_out_channels)

            self.fc_reg = nn.ModuleList()
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg.append(build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg))

    def forward_2fcs(self, x, stage, others=None):
        if self.num_shared_fcs > 0:
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x

    def forward_head(self, x, stage, others=None):
        x_cls, x_ins, x_reg = x, x, x
        cls_score = self.fc_cls[stage](x_cls) if self.with_cls else None
        ins_score = self.fc_ins[stage](x_ins) if self.with_ins else None
        reg_box = self.fc_reg[stage](x_reg) if self.with_distill and stage == self.num_stages - 2 else None
        # reg_box= None
        if self.with_others:
            others = dict(x_feat=x_cls)
            return cls_score, ins_score, reg_box, others
        else:
            return cls_score, ins_score, reg_box, None

    # def forward_objectness(self, x, stage, others=None):
    #     object_score = self.fc_objness(x) if stage == 1 else None
    #     return object_score

    def forward_iou(self, x, stage, others=None):
        iou_score = self.fc_iou(x) if stage == 1 else None
        return iou_score

    def forward_distill_2fcs(self, x, others=None):
        # x=self.DropBlock2D(x)
        if self.with_distill_2fcs:
            x = x.flatten(1)
            for fc in self.distill_fcs:
                x = self.relu(fc(x))
        else:
            if self.num_shared_fcs > 0:
                x = x.flatten(1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
        return x
    def forward_distill_2fcs_objness(self, x, others=None):
        # x=self.DropBlock2D(x)
        x = x.flatten(1)
        for fc in self.objness_2fcs:
            x = self.relu(fc(x))

        return x

    def forward_distill_reg_ori(self, x, others=None):
        bbox_pred = self.fc_distill_reg_ori(x)
        return bbox_pred

    def forward_distill_reg_re(self, x, others=None):
        bbox_pred = self.fc_distill_reg_re(x)
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

    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_mil(self,
                 stage_mode,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 labels,
                 reg_bboxes,
                 gt_bboxes,
                 label_weights,
                 retrain_weights,
                 reduction_override=None):
        losses = dict()

        if stage_mode == 'CBP':
            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    label_valid = proposals_valid_list
                    cls_score = cls_score.softmax(dim=-1)
                    num_sample = cls_score.shape[0]
                    pos_loss, bag_acc, num_pos = self.loss_mil1(
                        cls_score,
                        ins_score,
                        labels,
                        label_valid,
                        label_weights.unsqueeze(-1), )
                    pos_loss *= self.loss_p2b_weight  # 7/19 dididi
                    if isinstance(pos_loss, dict):
                        losses.update(pos_loss)
                    else:
                        losses['loss_instance_mil'] = pos_loss
                    losses['bag_acc'] = bag_acc

                    ## base loss when shake
                    base_loss = False

        elif stage_mode == 'PBR':
            if cls_score is not None:
                if cls_score.numel() > 0:
                    label_valid = proposals_valid_list
                    cls_score = cls_score.sigmoid()
                    num_sample = cls_score.shape[0]
                    pos_loss, bag_acc, num_pos = self.loss_mil2(
                        cls_score,
                        ins_score,
                        labels,
                        label_valid,
                        label_weights.unsqueeze(-1), )
                    if isinstance(pos_loss, dict):
                        losses.update(pos_loss)
                    else:
                        losses['loss_instance_mil'] = pos_loss
                    losses['bag_acc'] = bag_acc

            if neg_cls_score is not None:
                num_neg, num_class = neg_cls_score.shape
                neg_cls_score = neg_cls_score.sigmoid()
                neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                loss_weights = 0.75
                neg_valid = neg_weights.reshape(num_neg, -1)
                assert num_sample != 0
                neg_loss = self.loss_mil2.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
                neg_loss = loss_weights * label_weights.float().mean() * weight_reduce_loss(neg_loss, None,
                                                                                            avg_factor=num_sample)
                neg_loss *= self.loss_p2b_weight  # 7/19 dididi
                losses.update({"neg_loss": neg_loss})

        if gt_bboxes is not None and reg_bboxes is not None:
            # gt_bboxes  = gt_bboxes.unsqueeze(1).expand(reg_bboxes.shape)
            reg_bboxes = reg_bboxes.reshape(-1, 4)

            gt_bboxes = gt_bboxes.reshape(-1, 4)
            # loss = self.loss_bbox()
            num_reg_pos = reg_bboxes.shape[0]

            # reg_weight = label_weights_ * retrain_weights
            label_weights_ = retrain_weights.unsqueeze(-1).expand(cls_score.shape[:2])
            losses['loss_bbox'] = self.loss_bbox(
                reg_bboxes,
                gt_bboxes,
                label_weights_.reshape(-1, 1),
                avg_factor=num_reg_pos
            )

        return losses

    def loss_drag(self, bboxes_pred, gt_bboxes, label_weights, gt_labels):
        losses = dict()
        num_gt, num_ins = bboxes_pred.shape[:2]
        a = bboxes_pred[:, :, None, :].expand(num_gt, num_ins, num_gt, 4)
        b = gt_bboxes[None, None, :, :].expand(num_gt, num_ins, num_gt, 4)

        weights = b.new_full(b.shape, 1)
        weights *= (bbox_overlaps(a, b, is_aligned=True) > 0.3)[..., None]
        weights[range(num_gt), :, range(num_gt), :] = 0
        losses['loss_anti_iou'] = self.loss_anti_iou(a.reshape(-1, 4), b.reshape(-1, 4), weights.reshape(-1, 4),
                                                     avg_factor=weights[..., 0].sum())

        return losses

    def loss_objectness(self, pos_object_score, neg_object_score,iou_score, pos_target, neg_target,cascade_weight=None):
        """
        :param pos_object_score:
        :param neg_object_score:
        :param iou_score:
        :param pos_target: shape=(num_gt, samples_per_gt,1)
        :param neg_target:
        :return:
        """
        losses = dict()
        # (pos_target > 0.3)
        # if pos_object_score.shape[1] != 1:
        #     pos_target_ = pos_object_score.new_full(pos_object_score.shape, 1)
        #     pos_target_[:,:,0,:]=0
        #     # num_gt = pos_target_.shape[0]
        #     # pos_target_[range(num_gt), :, range(num_gt)] = 0
        #     # pos_target_ = pos_target.reshape(num_gt, -1, 1)[:, :, None, :] * pos_target_
        # object_score = torch.cat([pos_object_score.reshape(-1, 1), neg_object_score])
        #
        # target = torch.cat([pos_target_.reshape(-1), neg_target.reshape(-1) ]) # (num_gt*samples_per_gt*(1+(chosen_gt-1)))
        # # losses['loss_object_score'] = self.loss_obj(object_score.sigmoid(), ((target>0).long(),(target>0).float()),avg_factor=pos_target.shape[0])
        # losses['loss_object_score'] = self.loss_obj(object_score, target.long(), avg_factor=pos_target.shape[0])
        mean,std=0.5,0.5
        label_weights_ = cascade_weight.unsqueeze(-1).expand(iou_score.shape[:2])

        losses['loss_iou']=self.loss_iou(iou_score.reshape(-1),(pos_target.reshape(-1)-mean)/std)
        return losses
