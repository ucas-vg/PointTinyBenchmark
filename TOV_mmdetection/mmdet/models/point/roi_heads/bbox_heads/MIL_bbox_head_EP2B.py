import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import torch.nn.functional as F
from mmdet.models.losses import accuracy
from mmdet.models.losses.utils import weight_reduce_loss
# from mmdet.core.point.dcdet_utils.db import DropBlock2D

@HEADS.register_module()
class Shared2FCInstanceMILHeadEP2B(Shared2FCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 stage_modes=['CBP', 'PBR'],
                 num_stages=1,
                 num_shared_fcs=2,
                 num_cls_fcs=0,
                 num_reg_fcs=0,
                 num_ref_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 loss_type=None,
                 with_loss_pseudo=False,
                 with_others=False,
                 with_reg=False,
                 with_sem=False,
                 loss_mil1=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25),  # weight
                 loss_p2b_weight=1.0,  # 7/19 dididi
                 loss_mil2=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='fc_cls',
                         std=0.01,
                         bias_prob=0.01), ),
                 *args,
                 **kwargs):
        super(Shared2FCInstanceMILHeadEP2B, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_fcs + num_cls_fcs + num_reg_fcs > 0)

        self.with_reg = with_reg  ##################################

        if not self.with_cls:
            assert num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_fcs == 0
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.num_ref_fcs = num_ref_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.stage_modes = stage_modes
        self.loss_mil1 = build_loss(loss_mil1)
        self.loss_mil2 = build_loss(loss_mil2)
        self.loss_p2b_weight = loss_p2b_weight  # 7/19 dididi
        self.with_ins = self.with_cls
        self.num_stages = num_stages
        self.with_others = with_others
        self.loss_type = loss_type
        self.with_loss_pseudo = with_loss_pseudo
        # add shared convs and fcs
        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        self.cls_last_dim = self.ins_last_dim = self.reg_last_dim = last_layer_dim

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_cls_fcs == 0:
                self.ins_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        # add reg specific branch
        if self.with_reg:
            _, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(0, self.num_reg_fcs, self.shared_out_channels)
        if self.num_stages == 2:
            _, self.ref_fcs, self.ref_last_dim = \
                self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)
            if self.num_stages == 3:
                _, self.ref2_fcs, self.ref2_last_dim = \
                    self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed

        self.fc_cls = nn.ModuleList()
        self.fc_ins = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        for i in range(self.num_stages):
            if i < 1:
                num_cls = self.num_classes + 1
            else:
                num_cls = self.num_classes
            if self.with_cls:
                self.fc_cls.append(build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.cls_last_dim,
                    out_features=num_cls))
            if self.with_ins:
                self.fc_ins.append(build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.cls_last_dim,
                    out_features=num_cls))
            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
                self.fc_reg.append(build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=self.reg_last_dim,
                    out_features=out_dim_reg))

        self.with_sem=with_sem

        # self.cls_wise_feature = nn.Embedding(self.num_classes, 256)
        # cls_wise_feature = self.cls_wise_feature.weight.clone()

    def forward(self, x, stage, others=None):
        # shared part
        # if self.with_sem:
        #     x = self.DropBlock2D(x)
        if stage == 0 or (stage >= 1 and self.num_ref_fcs == 0):
            if self.num_shared_fcs > 0:
                x = x.flatten(1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
                x_cls, x_ins, x_reg = x, x, x

        elif stage >= 1 and self.num_ref_fcs > 0:
            x_ref = x
            x_ref = x_ref.flatten(1)
            for fc in self.ref_fcs:
                x_ref = self.relu(fc(x_ref))
            x_cls, x_ins, x_reg = x_ref, x_ref, x_ref

        if self.with_reg:
            for fc in self.reg_fcs:
                if x_reg.dim() > 2:
                    x_reg = x_reg.flatten(1)
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls[stage](x_cls) if self.with_cls else None
        ins_score = self.fc_ins[stage](x_ins) if self.with_ins else None
        reg_box = self.fc_reg[stage](x_reg) if self.with_reg and stage == self.num_stages - 2 else None
        # reg_box= None
        if self.with_others:
            others = dict(x_feat=x_cls)
            return cls_score, ins_score, reg_box, others
        else:
            return cls_score, ins_score, reg_box, None

    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_mil(self,
                 stage_mode,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 reg_bboxes,
                 labels,
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
                    if base_loss:
                        num_class = cls_score.shape[-1]
                        base_cls_score = cls_score.reshape(num_sample, -1, 5, num_class)[:, :, 0, :]
                        base_ins_score = ins_score.reshape(num_sample, -1, 5, num_class)[:, :, 0, :]
                        base_label_valid = label_valid.reshape(num_sample, -1, 5, 1)[:, :, 0, :]
                        base_pos_loss, _, _ = self.loss_mil1(
                            base_cls_score,
                            base_ins_score,
                            labels,
                            base_label_valid,
                            label_weights.unsqueeze(-1), )
                        base_weight = 0.5
                        losses['loss_base_mil'] = base_pos_loss * base_weight
                if neg_cls_score is not None:
                    num_neg, num_class = neg_cls_score.shape
                    neg_cls_score = neg_cls_score.softmax(dim=-1)
                    neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                    neg_labels[:, -1] = 1
                    loss_weights = 0.25
                    neg_valid = neg_weights.reshape(num_neg, -1)
                    # assert num_sample != 0
                    neg_cls_score = neg_cls_score.clamp(0, 1)
                    neg_loss = F.binary_cross_entropy(neg_cls_score, neg_labels, neg_valid.float(), reduction="none")
                    # neg_valid.float())
                    neg_loss = loss_weights * weight_reduce_loss(neg_loss, None, avg_factor=neg_cls_score.shape[0])
                    losses.update({"neg_loss": neg_loss})
        elif stage_mode == 'PBR':
            if stage_mode == 'PBR': mode = 'mil-2'
            if mode == 'mil-2':
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

                if self.with_loss_pseudo:  ##pseudo_box loss
                    cls_score_pseudo = cls_score[:, 0]
                    label_valid = cls_score.new_full(
                        (cls_score.shape[0], 1), 1, dtype=torch.long)
                    labels_ = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
                    pseudo_loss = self.loss_mil2.gfocal_loss(cls_score_pseudo, labels_, label_valid.float())
                    loss_weights = 0.125
                    pseudo_loss = loss_weights * weight_reduce_loss(pseudo_loss, label_weights, avg_factor=num_sample)
                    losses.update({"pseudo_box_loss": pseudo_loss})
            elif mode == 'cluster cam':
                if cls_score is not None:
                    cls_score = cls_score.sigmoid()
                    num_sample = cls_score.shape[0]
                    # label_valid = cls_score.new_full(
                    #     (cls_score.shape[0], 1), 1, dtype=torch.long)
                    label_valid = proposals_valid_list
                    cls_score = cls_score.mean(dim=1)
                    labels_ = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
                    cluster_loss = self.loss_mil2.gfocal_loss(cls_score, labels_, label_valid.float())
                    loss_weights = 0.25
                    cluster_loss = loss_weights * weight_reduce_loss(cluster_loss, label_weights, avg_factor=num_sample)
                    losses.update({"cluster_loss": cluster_loss})
                    acc = accuracy(cls_score, labels)
                    losses['bag_acc'] = acc
            elif mode == 'mil-1':  ## same as PBR
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
            # if retrain_weights is not None:
            #     if cls_score is not None:
            #         cls_score = cls_score.sigmoid()
            #         cls_score_ = cls_score.reshape(-1, cls_score.shape[-1])
            #         labels_ = (retrain_weights == 0) * self.num_classes + (retrain_weights > 0) * labels.unsqueeze(-1)
            #         labels_ = labels_.reshape(-1)
            #         ## *3 because here is 0.25,but neg weight is 0.75
            #         # label_weights_ = (retrain_weights > 0) * label_weights.unsqueeze(-1).expand(cls_score.shape[:2])
            #
            #         label_weights_ = (retrain_weights == 0) * label_weights.mean() * 3 + (
            #                 retrain_weights > 0) * label_weights.unsqueeze(-1).expand(cls_score.shape[:2])
            #
            #         num_sample = retrain_weights.sum()
            #         label_valid = cls_score.new_full((cls_score_.shape[0], 1), 1)
            #         _labels_ = _expand_onehot_labels(labels_, None, cls_score.shape[-1])[0].float()
            #         cluster_loss = self.loss_mil2.gfocal_loss(cls_score_, _labels_, label_weights_.reshape(-1, 1))
            #         loss_weights = 0.25
            #         cluster_loss = loss_weights * weight_reduce_loss(cluster_loss, None,
            #                                                          avg_factor=num_sample)
            #         losses.update({"loss_retrain_cls": cluster_loss})
            #         acc = accuracy(cls_score_, labels_)
            #         losses['bag_acc'] = acc
        if gt_bboxes is not None and reg_bboxes is not None:
            # gt_bboxes  = gt_bboxes.unsqueeze(1).expand(reg_bboxes.shape)
            reg_bboxes = reg_bboxes.reshape(-1, 4)

            gt_bboxes = gt_bboxes.reshape(-1, 4)
            # loss = self.loss_bbox()
            num_reg_pos = reg_bboxes.shape[0]

            # reg_weight = label_weights_ * retrain_weights
            label_weights_ =  retrain_weights.unsqueeze(-1).expand(cls_score.shape[:2])
            losses['loss_bbox'] = self.loss_bbox(
                reg_bboxes,
                gt_bboxes,
                label_weights_.reshape(-1, 1),
                avg_factor=num_reg_pos
            )
                # avg_factor=((num_reg_pos + 1e-5) ** 2 / num_reg_pos))
        return losses
