import warnings

from mmcv.runner import BaseModule
import torch.nn as nn
from mmdet.models.losses import accuracy
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.core.point.cpr_utils.utils import *
from mmdet.core import (multi_apply)
from mmdet.models.builder import HEADS


def one_hot(gt_labels, num_class):
    """
        gt_labels: (...,)
    Return:
        gt_labels_on_hot: (..., num_class)
    """
    origin_shape = gt_labels.shape
    gt_labels = gt_labels.reshape(-1)
    gt_labels_one_hot = torch.full((len(gt_labels), num_class), 0., dtype=torch.float32).to(gt_labels.device)
    gt_labels_one_hot[torch.arange(len(gt_labels)), gt_labels] = 1
    return gt_labels_one_hot.reshape(*origin_shape, num_class)


def gfocal_loss(p, q, w=1.0, eps=1e-6):
    l1 = (p - q) ** 2
    l2 = q * (p + eps).log() + (1 - q) * (1 - p + eps).log()
    return -(l1 * l2 * w).sum(dim=-1)


def bce_loss(p, q, w=1.0, eps=1e-6):
    # for q == 1 (pos),
    #    when p < eps, (too bad) limited grad not too large as 1/(eps+p) in (1/2/eps, 1/eps)
    #    when p > 1 - eps, (enough good) set grad as 1/(eps+p) in (1/(1+eps), 1/eps)
    # for q == 0 (neg), when p > 1 - eps
    #    when p < eps, (enough good) set grad as 1/(1-p + eps) in (1/(1+eps), 1/eps)
    #    when p > 1 - eps, (too bad) limited grad not too large as 1/(1-p + eps) in (1/2/eps, 1/eps)
    l2 = q * (p + eps).log() + (1 - q) * (1 - p + eps).log()
    # print(p.min().item(), p.max().item())
    return -(l2 * w).sum(dim=-1)


class MILLoss(object):
    def __init__(self, avg_neg_ins=None, type='gfocal_loss', debug_cfg=dict(), out_bg_cls=False):
        self.avg_neg_ins = avg_neg_ins
        self.loss = eval(type)
        self.statistic_obj = Statistic()
        self.debug_cfg = debug_cfg
        self.out_bg_cls = out_bg_cls

    def get_avg_neg_ins_prob(self, prob_ins, labels_onehot):
        num_ins, num_class = prob_ins.shape[-2:]
        pos_position = labels_onehot.unsqueeze(dim=-2).repeat([1]*(len(labels_onehot.shape)-1) + [num_ins, 1]) > 0
        prob_ins_new = torch.ones(prob_ins.shape, dtype=prob_ins.dtype).to(prob_ins.device) / num_ins
        prob_ins_new[pos_position] = prob_ins[pos_position]
        prob_ins = prob_ins_new
        return prob_ins

    def get_instance_full_loss(self, prob_cls, labels_onehot, label_weights):
        """
            given the label of bag to each instance in bag to calculate loss
        """
        num_ins, num_class = prob_cls.shape[-2:]
        repeat = [1] * (len(labels_onehot.shape) - 1) + [num_ins, 1]
        labels = labels_onehot.unsqueeze(dim=-2).repeat(repeat)  # (..., num_ins, num_class)
        label_weights = label_weights.unsqueeze(dim=-2).repeat(repeat)  # (..., num_ins, num_class)
        loss = self.loss(prob_cls, labels, label_weights)  # (..., num_ins)
        loss = loss.sum(dim=-1) / num_ins
        return loss  # (...,)

    def get_mil_pos_instance_neg_loss(self, prob_cls, prob, labels_onehot, label_weights):
        pos_label_weights = label_weights * labels_onehot
        neg_label_weights = label_weights * (1 - labels_onehot)
        mil_pos_loss = self.loss(prob, labels_onehot, pos_label_weights)
        instance_neg_loss = self.get_instance_full_loss(prob_cls, labels_onehot, neg_label_weights)
        return mil_pos_loss + instance_neg_loss

    def statistic(self, prob_cls, prob_ins, valid, gt_label, num_lvl, name_prefix, gt_true_bboxes):
        if self.debug_cfg.get('print_max_lvl', False):
            topk = 3
            prob = prob_cls * prob_ins
            prob_cls = (prob_cls*valid)[torch.arange(len(gt_label)), ..., gt_label]
            prob = prob[torch.arange(len(gt_label)), ..., gt_label]
            # shape = prob_cls.shape

            def sta_max(p):
                p = p.flatten(0, -2)
                n, num_chosen_all_lvl = p.shape
                p = p.reshape(n, num_lvl, -1)
                max_v, max_lvl = p.sort(dim=-1)[0][..., -topk:].mean(dim=-1).max(dim=-1)
                return max_lvl

            true_bboxes = torch.cat(gt_true_bboxes, dim=0)
            wh = true_bboxes[:, 2:] - true_bboxes[:, :2]
            size = np.log2(((wh[:, 0] * wh[:, 1]) ** 0.5).detach().cpu().numpy())

            def sta_size(max_lvl):
                size_lvl = [[] for l in range(num_lvl)]
                for i, l in enumerate(max_lvl):
                    size_lvl[l].append(size[i])
                import numpy as np
                return size_lvl

            max_cls_lvl, max_lvl = sta_max(prob_cls), sta_max(prob)
            self.statistic_obj.print_count(f"[{name_prefix} max lvl of {topk}-mean (prob_cls)]",
                                           max_cls_lvl, momentum=0.99, log_interval=500)
            self.statistic_obj.print_c_ema(f"[{name_prefix} log2 size of gt (prob cls)]",
                                         sta_size(max_cls_lvl), momentum=0.99, log_interval=500)
            self.statistic_obj.print_count(f"[{name_prefix} max lvl of {topk}-mean (prob)]",
                                           max_lvl, momentum=0.99, log_interval=500)
            self.statistic_obj.print_c_ema(f"[{name_prefix} log2 size of gt (prob)]",
                                         sta_size(max_lvl), momentum=0.99, log_interval=500)

    def __call__(self, prob_cls, prob_ins, valid, labels, weights=1, is_one_hot=False, avg_factor=None):
        """
            prob_cls: (..., num_ins, num_class)
            prob_ins: (..., num_ins, num_class)
            valid:    (..., num_ins, 1/num_class)
            labels:   (...)  / (..., num_class)
            weights:  (..., 1/num_class) / float
        """
        labels_onehot = labels if is_one_hot else one_hot(labels, prob_cls.shape[-1])  # (..., num_class)
        if self.avg_neg_ins is not None:
            prob_ins = self.get_avg_neg_ins_prob(prob_ins, labels_onehot)

        valid = valid.float()
        prob_ins = F.normalize(prob_ins * valid, dim=-2, p=1)      # user should make sure no inf by div 0 here
        prob = prob_cls * prob_ins
        prob = prob.sum(dim=-2)                   # (..., num_class)
        # print(prob.shape, labels.shape)
        acc = accuracy(prob[..., :-1] if self.out_bg_cls else prob, labels)  # not include bg for bag acc

        if isinstance(valid, (int, float)):
            label_weights = 1.0
            num_sample = len(prob)
        else:  # isinstance(valid, torch.Tensor):
            label_weights = (valid.sum(dim=-2) > 0).float()
            num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)
        label_weights = label_weights * weights  # (..., num_class)

        if self.avg_neg_ins is None or self.avg_neg_ins == 'prob':
            loss = self.loss(prob, labels_onehot, label_weights)
        elif self.avg_neg_ins == 'loss':
            loss = self.get_mil_pos_instance_neg_loss(prob_cls, prob, labels_onehot, label_weights)
        else:
            raise ValueError

        if avg_factor is None:
            avg_factor = num_sample
        loss = weight_reduce_loss(loss, None, avg_factor=avg_factor)
        return loss, acc, num_sample


# class MultiDimMILHead(BaseModule):
#     def __init__(self, num_classes, in_channel,
#                  num_mil_dim=2,
#                  init_cfg=dict(
#                      type='Normal',
#                      layer=['Conv2d', 'Linear'],
#                      std=0.01,
#                      override=dict(
#                          type='Normal',
#                          name='cls_out',
#                          std=0.01,
#                          bias_prob=0.01),
#                  )):
#         self.num_mil_dim = num_mil_dim
#         self.in_channel = in_channel
#         self.num_cls_out = num_classes
#
#         # classification branch and selection branch
#         chn = self.in_channel
#         self.cls_out = nn.Linear(chn, self.num_cls_out)
#         self.sel_outs = nn.ModuleList()
#         for i in range(self.num_mil_dim):
#             self.sel_outs.append(nn.Linear(chn, self.num_cls_out))
#
#         super(MultiDimMILHead, self).__init__(init_cfg)
#
#     def forward_single_MIL(self, feat, cls_prob, sel_out_fc):
#         feat_shape = feat.shape                                 # (n, dk, ...di, C)
#         sel_out = sel_out_fc(feat.reshape(-1, feat_shape[-1]))  # (n*dk*d(i+1)*di, num_class)
#         sel_out = sel_out.reshape(*feat_shape[:-1], -1)         # (n, dk, ...d(i+1), di, num_class)
#         sel_prob = sel_out.softmax(dim=-2)                      # (n, dk, ...d(i+1), di, num_class)
#         cls_prob = (cls_prob * sel_prob)                        # (n, dk, ...d(i+1), di, num_class)
#         return cls_prob
#
#     def get_feat_weight(self, cls_prob, gt_labels):
#         """
#             cls_prob: shape=(n, dk, ...d(i+1), di, num_class)
#             gt_labels: shape=(n,)
#         """
#         # feat weight in dim=-2 must sum equal as 1, cause it is a weighted avg pooling in dim=-2
#         idx = torch.arange(len(cls_prob)).to(cls_prob.device)
#         feat_weight = cls_prob[idx, ..., gt_labels].unsqueeze(dim=-1)  # gt_cls_prob
#         feat_weight = feat_weight / feat_weight.sum(dim=-2, keepdim=True)
#         return feat_weight
#
#     def forward(self, feat, gt_labels):
#         """
#             feat: shape=(n, d1, d0, num_feat_channel)
#             gt_labels: shape=(n)
#             sel_outs[0] for (dim=-2)d0, sel_outs[1] for (dim=-3)d1, ...
#         """
#         feat_shape = feat.shape                                             # (n, dk, ...d1, d0, C)
#         cls_out = self.cls_out(feat.reshape(-1, feat_shape[-1]))
#         cls_out = cls_out.reshape(*feat_shape[:-1], -1)
#         cls_prob = cls_out.sigmoid()                                        # (n, dk, ...d1, d0, num_class)
#
#         cls_probs = [cls_prob]
#         for i, sel_out_fc in enumerate(self.sel_outs):
#             cls_prob = self.forward_single_MIL(feat, cls_prob, sel_out_fc)  # (n, dk, ...d(i+1), di, num_class)
#             # [(n, dk, ...d(i+1), di, 1) * (n, dk, ...d(i+1), di, C)].sum(dim=-2)
#             feat_weight = self.get_feat_weight(cls_prob, gt_labels)
#             feat = (feat * feat_weight).sum(dim=-2)                         # (n, dk, ...d(i+1), C)
#             cls_prob = cls_prob.sum(dim=-2)
#             cls_probs.append(cls_prob)
#         return cls_probs
#
#     def estimate(self, feat, gt_labels):
#         cls_probs = self.forward(feat, gt_labels)[:-1]
#         idx = torch.arange(len(gt_labels)).to(gt_labels.device)
#         estimate_res = []
#         for i in range(len(self.sel_outs)):
#             cls_prob = cls_probs[-1][idx, ..., gt_labels]    # (n, dk)
#             best_idx = cls_prob.argmax(dim=-1)               # (n)
#             cls_probs = [cls_prob[idx, best_idx] for cls_prob in cls_probs[:-1]]
#             estimate_res.append(best_idx)
#         estimate_res = torch.stack(estimate_res, dim=-1)
#         return estimate_res  # (n, k)


class BaseMILHead(BaseModule):
    def __init__(self,
                 num_classes,

                 sel_share_head_conv=True,
                 loss_cfg=dict(
                     mil_loss=dict()
                 ),

                 out_bg_cls=False,          # does last branch need out logit for background
                 prob_cls_type='sigmoid',   # 'sigmoid', 'softmax', 'normed_sigmoid'
                 normed_sigmoid_p=1,        # p for prob_cls_type=='normed_sigmoid'
                 debug_cfg=dict(),
                 init_cfg=None):
        assert (prob_cls_type == 'softmax' and out_bg_cls) or (prob_cls_type != 'softmax' and not out_bg_cls), ""

        self.out_bg_cls = out_bg_cls
        self.num_cls_out = num_classes + 1 if out_bg_cls else num_classes
        self.prob_cls_type = prob_cls_type
        self.normed_sigmoid_p = normed_sigmoid_p

        self.loss_cfg = loss_cfg

        base_loss = loss_cfg.get('loss_type', 'gfocal_loss')
        mil_loss = loss_cfg.get("mil_loss", {})
        if 'type' not in mil_loss:
            mil_loss['type'] = base_loss
        mil_loss['debug_cfg'] = debug_cfg
        mil_loss['out_bg_cls'] = out_bg_cls
        self.mil_loss = MILLoss(**mil_loss)
        self.base_loss = eval(base_loss)  # focal loss or bce loss

        self.sel_share_head_conv = sel_share_head_conv
        super(BaseMILHead, self).__init__(init_cfg)
        self.deprecated_check()

    def deprecated_check(self):
        if 'with_gt_loss' in self.loss_cfg:
            warnings.warn("[DeprecationWarning]: using with_ann_loss instead of with_gt_loss.")
            self.loss_cfg['with_ann_loss'] = self.loss_cfg.pop('with_gt_loss')
        if 'gt_loss_weight' in self.loss_cfg:
            warnings.warn("[DeprecationWarning]: using ann_loss_weight instead of gt_loss_weight.")
            self.loss_cfg['ann_loss_weight'] = self.loss_cfg.pop('gt_loss_weight')

    def random_remove(self, *all_pts, random_remove_rate=0):
            """
            all_pts: [k, (..., a)]
            Returns:
            """
            if random_remove_rate > 0:
                for pts in all_pts:
                    valid = pts[..., -1]
                    remove = torch.rand(valid.shape) < random_remove_rate
                    valid[remove] = 0.
                    pts[..., -1] = valid

    def get_cls_prob(self, cls_out):
        """
        Args:
            cls_out: (..., C*K),
        Returns:
        """
        prob_cls_type = self.prob_cls_type
        shape = cls_out.shape[:-1]
        cls_out = cls_out.reshape(*shape, self.num_cls_out, -1)
        if prob_cls_type == 'softmax':
            prob_cls = cls_out.softmax(dim=-2)
        elif prob_cls_type == 'sigmoid':
            prob_cls = cls_out.sigmoid()
        elif prob_cls_type == 'normed_sigmoid':
            prob_cls = cls_out.sigmoid()
            prob_cls = F.normalize(prob_cls, p=self.normed_sigmoid_p, dim=-2)
        else:
            raise ValueError()
        return prob_cls.reshape(*shape, -1)

    def loss(self, pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes, gt_weights):
        """
        Args:
            pos_data.cls_feats: [num_lvl, (num_gts, num_refine, num_chosen, num_feat_c)]
            pos_data.ins_feats: [num_lvl, (num_gts, num_refine, num_chosen, num_feat_c)]
            pos_data.pts: [num_lvl, (num_gts, num_refine, num_chosen, 3)]
            pos_data.valid: [num_lvl, (num_gts, num_refine, num_chosen, num_class)]
            neg_data.cls_feats: [num_lvl, (num_negs, num_feat_c)]
            neg_data.pts:  [num_lvl, (num_negs, 3)]
            neg_data.valid: [num_lvl, (num_negs, num_class)]
            gt_labels_all: (num_gts,)
            gt_r_points: [B, (num_gt_per_img, num_refine, 2)]
            gt_true_bboxes:
            gt_weights: (num_gts, )
        Returns:
        """
        raise NotImplementedError

    def predict(self, bag_data, grid_data, gt_labels):
        raise NotImplementedError


@HEADS.register_module()
class BasicMILHead(BaseMILHead):
    def __init__(self, num_classes, in_channel,
                 # head_feat_cfg
                 stacked_fcs=0,             # number of fc in fc_head
                 fc_out_channels=1024,      # channel of fc in fc_head
                 sel_share_head_conv=True,  # support for old api
                 sel_share_head_fc=True,
                 sel_share_head_out=False,  # does selection branch share logit with classify branch
                 binary_ins=False,

                 out_bg_cls=False,          # does last branch need out logit for background
                 prob_cls_type='sigmoid',   # 'sigmoid', 'softmax', 'normed_sigmoid'
                 normed_sigmoid_p=1,        # p for prob_cls_type=='normed_sigmoid'

                 # 3. loss
                 loss_cfg=dict(
                     with_bag_loss=True,
                     bag_loss_weight=0.25,
                     refine_bag_policy='only_refine_bag',
                     random_remove_rate=0.4,
                     neg_sampler='all',    # optional: "all", "ohem"
                     neg_sample_rate=-1.,  # -1 means use all neg, > 0 means num_neg = num_pos *neg_sample_rate
                     with_neg_loss=True,
                     neg_loss_weight=0.75,
                     with_ann_loss=True,
                     ann_loss_weight=0.125,
                     loss_type='gfocal_loss',
                     mil_loss=dict(),
                 ),

                 debug_cfg=dict(),
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_out',
                         std=0.01,
                         bias_prob=0.01),
                 ),
                 ):
        self.in_channel = in_channel
        # head_feat_cfg
        self.stacked_fcs = stacked_fcs
        self.fc_out_channels = fc_out_channels
        self.sel_share_head_fc = sel_share_head_fc
        self.sel_share_head_out = sel_share_head_out

        self.binary_ins = binary_ins

        # if prob_cls_type == 'softmax':
        #     init_cfg.pop('override')

        super(BasicMILHead, self).__init__(
            num_classes, sel_share_head_conv, loss_cfg, out_bg_cls, prob_cls_type, normed_sigmoid_p, debug_cfg, init_cfg
        )

        self._init_layers()

    def _init_layers(self):
        chn = self.in_channel
        self.cls_fcs = nn.ModuleList()
        self.ins_fcs = nn.ModuleList()
        for i in range(self.stacked_fcs):
            self.cls_fcs.append(nn.Linear(chn, self.fc_out_channels))  # ReLU used in invoking-time
            if not self.sel_share_head_fc:
                self.ins_fcs.append(nn.Linear(chn, self.fc_out_channels))
            chn = self.fc_out_channels

        self.cls_out = nn.Linear(chn, self.num_cls_out)
        if not self.sel_share_head_out:
            num_ins_out = self.num_cls_out * 2 if self.binary_ins else self.num_cls_out
            self.ins_out = nn.Linear(chn, num_ins_out)
        else:
            assert not self.binary_ins
            self.ins_out = self.cls_out

    def forward_with_head_fc_and_out(self, pts_cls_feats, pts_ins_feats=None):
    # def get_pts_outs(self, pts_cls_feats, pts_ins_feats=None):
        """
        Args:
            pts_cls_feats: [num_lvl, (..., C)]
            pts_ins_feats: [num_lvl, (..., C]
            ins_same_as_cls:
        Returns:
            cls_outs: [num_lvl, (..., num_class)]
            ins_outs: [num_lvl, (..., num_class)]
        """

        def forward_with_fc(feat, fcs):
            feat = feat.flatten(0, -2)
            for i, fc in enumerate(fcs):
                feat = self.relu(fc(feat))
            return feat

        def get_outs_single(cls_f, ins_f=None):
            shape = cls_f.shape
            cls_f = forward_with_fc(cls_f, self.cls_fcs)
            cls_o = self.cls_out(cls_f).reshape(*shape[:-1], -1)

            if ins_f is None:
                return cls_o,
            sel_share_head_fc = self.sel_share_head_fc and self.sel_share_head_conv
            ins_f = forward_with_fc(ins_f, self.ins_fcs) if not sel_share_head_fc else cls_f
            ins_o = self.ins_out(ins_f).reshape(*shape[:-1], -1) if not \
                (sel_share_head_fc and self.sel_share_head_out) else cls_o
            return cls_o, ins_o

        if pts_ins_feats is None:
            cls_outs, = multi_apply(get_outs_single, pts_cls_feats)
            return cls_outs
        else:
            cls_outs, ins_outs = multi_apply(get_outs_single, pts_cls_feats, pts_ins_feats)
            return cls_outs, ins_outs

    def loss(self, pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes, gt_weights,
             img_metas, cls_feat_shape):
        pos_data.cls_outs, pos_data.ins_outs = self.forward_with_head_fc_and_out(pos_data.cls_feats, pos_data.ins_feats)
        neg_data.cls_outs = self.forward_with_head_fc_and_out(neg_data.cls_feats)
        return self.loss0(pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes, gt_weights)
                          # ,img_metas, cls_feat_shape)

    def predict(self, bag_data, grid_data, gt_labels):
        bag_data.cls_outs, bag_data.ins_outs = self.forward_with_head_fc_and_out(bag_data.cls_feats, bag_data.ins_feats)
        grid_data.cls_outs = self.forward_with_head_fc_and_out(grid_data.cls_feats)
        bag_data.cls_prob = [self.get_cls_prob(bag_cls_feat_lvl) for bag_cls_feat_lvl in bag_data.cls_outs]
        grid_data.cls_prob = [self.get_cls_prob(grid_cls_outs_lvl) for grid_cls_outs_lvl in grid_data.cls_outs]
        return bag_data, grid_data

    def ann_loss(self, gt_cls_prob, gt_ins_outs, gt_valid, gt_labels_all, gt_weights,
                 gt_true_bboxes):
        num_gts, num_refine, num_lvl, _ = gt_valid.shape
        gt_type = self.loss_cfg.get('gt_type', 'gt_refine')

        if gt_type == 'gt_refine':
            gt_labels_rep = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
            gt_valid = gt_valid.reshape(num_gts * num_refine, num_lvl, -1)
            gt_weights_rep = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
            gt_weights_rep = gt_valid.float() * gt_weights_rep.reshape(-1, 1, 1)
            gt_cls_prob = gt_cls_prob.reshape(num_gts*num_refine, num_lvl, -1)
            gt_ins_outs = gt_ins_outs.reshape(num_gts*num_refine, num_lvl, -1)
        elif gt_type == 'gt':
            gt_labels_rep = gt_labels_all
            gt_valid = gt_valid[:, 0].float()
            gt_weights_rep = gt_valid * gt_weights.reshape(-1, 1, 1)
            gt_cls_prob = gt_cls_prob.reshape(num_gts, num_refine, num_lvl, -1)[:, 0]
            gt_ins_outs = gt_ins_outs.reshape(num_gts, num_refine, num_lvl, -1)[:, 0]
        else:
            raise ValueError()

        gt_loss_type = self.loss_cfg.get('gt_loss_type', 'lvl_pos')
        if gt_loss_type == 'lvl_pos':
            gt_labels_rep = gt_labels_rep.unsqueeze(dim=1).repeat(1, num_lvl).flatten()
            gt_weights_rep = gt_weights_rep.flatten(0, 1)  # (x, num_lvl, num_class) => (x*num_lvl, num_class)
            gt_cls_prob = gt_cls_prob.flatten(0, 1)   # (x, num_lvl, num_class) => (x*num_lvl, num_class)

            gt_labels = torch.full(gt_cls_prob.shape, 0., dtype=torch.float32).to(gt_cls_prob.device)
            gt_labels[torch.arange(len(gt_labels)), gt_labels_rep] = 1
            num_pos = max((gt_weights_rep > 0).sum(), 1)
            gt_loss = self.base_loss(gt_cls_prob, gt_labels, gt_weights_rep)
            gt_loss = weight_reduce_loss(gt_loss, None, avg_factor=num_pos)
        elif gt_loss_type == 'lvl_mil':
            gt_ins_prob = gt_ins_outs.softmax(dim=-2)
            gt_ins_prob = F.normalize(gt_ins_prob * gt_valid, p=1, dim=-2)
            gt_weights = gt_weights.unsqueeze(dim=-1)
            self.mil_loss.statistic(gt_cls_prob, gt_ins_prob, gt_valid, gt_labels_rep, num_lvl, "gt",
                                    gt_true_bboxes)
            gt_loss, _, num_pos = self.mil_loss(gt_cls_prob, gt_ins_prob, gt_valid, gt_labels_rep, gt_weights)
        else:
            raise ValueError
        return gt_loss, num_pos

    def neg_loss(self, neg_cls_outs, neg_valid, num_pos):
        neg_prob = self.get_cls_prob(neg_cls_outs)
        num_neg, num_class = neg_prob.shape
        neg_labels = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_prob.device)
        neg_valid = neg_valid.reshape(num_neg, -1)
        if self.out_bg_cls:  # last dim as background,
            neg_labels[:, -1] = 1
            neg_valid = torch.cat([neg_valid, neg_valid.all(dim=-1).unsqueeze(dim=-1)], dim=-1)
        neg_loss = self.base_loss(neg_prob, neg_labels, neg_valid.float())

        if self.loss_cfg.get('neg_sampler', 'all') == 'ohem' and self.loss_cfg.get('neg_sample_rate', -1.0) > 0:
            num_neg = int(round(num_pos * self.loss_cfg['neg_sample_rate']))
            neg_loss = neg_loss.topk(num_neg)[0]
        return weight_reduce_loss(neg_loss, None, avg_factor=num_pos)

    def bag_loss(self, pos_cls_outs, pos_ins_outs, pos_valid, gt_labels_all, gt_weights,
                 num_lvl, gt_true_boxes):
        num_gts, num_refine, num_chosen, _ = pos_valid.shape
        refine_bag_policy = self.loss_cfg["refine_bag_policy"]
        if refine_bag_policy == 'independent_with_gt_bag':
            # treat num_refine as independent bags
            # pos_pts = pos_pts.reshape(num_gts * num_refine, num_chosen, 3)
            pos_cls_outs = pos_cls_outs.reshape(num_gts * num_refine, num_chosen, -1)
            pos_ins_outs = pos_ins_outs.reshape(num_gts * num_refine, num_chosen, -1)
            pos_valid = pos_valid.reshape(num_gts * num_refine, num_chosen, -1)
            pos_weights = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
            gt_labels_all = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
        elif refine_bag_policy == 'merge_to_gt_bag':
            # pos_pts = pos_pts.reshape(num_gts, num_refine * num_chosen, 3)
            pos_cls_outs = pos_cls_outs.reshape(num_gts, num_refine * num_chosen, -1)
            pos_ins_outs = pos_ins_outs.reshape(num_gts, num_refine * num_chosen, -1)
            pos_valid = pos_valid.reshape(num_gts, num_refine * num_chosen, -1)
            pos_weights = gt_weights
        elif refine_bag_policy == 'only_refine_bag':
            si = 1 if num_refine > 1 else 0
            # pos_pts = pos_pts[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, 3)
            pos_cls_outs = pos_cls_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
            pos_ins_outs = pos_ins_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
            pos_valid = pos_valid[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
            pos_weights = gt_weights
        else:
            raise ValueError
        pos_weights = pos_weights.reshape(-1, 1)
        pos_cls_prob = self.get_cls_prob(pos_cls_outs)
        pos_ins_prob = pos_ins_outs.softmax(dim=1)
        self.mil_loss.statistic(pos_cls_prob, pos_ins_prob, pos_valid, gt_labels_all, num_lvl, "bag",
                                gt_true_boxes)
        pos_loss, bag_acc, num_pos = self.mil_loss(pos_cls_prob, pos_ins_prob, pos_valid, gt_labels_all, pos_weights)
        return pos_loss, bag_acc, num_pos

    def loss0(self, pos_data: PtAndFeat, neg_data: PtAndFeat, gt_labels_all, gt_r_points,
              gt_true_bboxes=None, gt_weights=None,
              # img_metas=None, cls_feat_shape=None):
              ):
        """
        Args:
            pos_data.cls_outs: [num_lvl, (num_gts, num_refine, num_chosen, num_class)], 2 is cls and ins
            pos_data.ins_outs: [num_lvl, (num_gts, num_refine, num_chosen, num_class)]
            pos_data.pts: [num_lvl, (num_gts, num_refine, num_chosen, 3)]
            pos_data.valid: [num_lvl, (num_gts, num_refine, num_chosen, num_class)]
            neg_data.cls_outs: [num_lvl, (num_negs, num_class)]
            neg_data.pts:  [num_lvl, (num_negs, 3)]
            neg_data.valid: [num_lvl, (num_negs, num_class)]
            gt_labels_all: (num_gts,)
            gt_r_points: [B, (num_gt_per_img, num_refine, 2)]
            gt_true_bboxes:
            gt_weights: (num_gts, )
        Returns:
        """
        # [num_lvl, (num_gts, num_refine, num_chosen, num_class)]=>(num_gts, num_refine, num_lvl, num_chosen, num_class)
        pos_cls_outs, pos_ins_outs = torch.stack(pos_data.cls_outs, dim=2), torch.stack(pos_data.ins_outs, dim=2)
        pos_pts, pos_valid = torch.stack(pos_data.pts, dim=2), torch.stack(pos_data.valid, dim=2)
        #
        neg_cls_outs, neg_pts = torch.cat(neg_data.cls_outs, dim=0), torch.cat(neg_data.pts, dim=0)
        neg_valid = torch.cat(neg_data.valid, dim=0)

        losses = {}
        num_gts, num_refine, num_lvl, num_chosen, _ = pos_pts.shape
        if self.loss_cfg.get('with_ann_loss', False):
            gt_valid = pos_valid[..., -1, :].reshape(num_gts, num_refine, num_lvl, -1)
            gt_cls_outs = pos_cls_outs[..., -1, :].reshape(num_gts, num_refine, num_lvl, -1)
            gt_ins_outs = pos_ins_outs[..., -1, :].reshape(num_gts, num_refine, num_lvl, -1)
            gt_cls_prob = self.get_cls_prob(gt_cls_outs)
            ann_loss, num_pos = self.ann_loss(gt_cls_prob, gt_ins_outs, gt_valid, gt_labels_all, gt_weights,
                                              gt_true_bboxes)
            losses['ann_loss'] = self.loss_cfg['ann_loss_weight'] * ann_loss

        if self.loss_cfg.get('with_bag_loss', True):
            self.random_remove(pos_valid, self.loss_cfg['random_remove_rate'])
            bag_loss_type = self.loss_cfg.get("bag_loss_type", "all_mil")
            if bag_loss_type == 'all_mil':
                pos_cls_outs = pos_cls_outs.reshape(num_gts, num_refine, num_lvl*num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts, num_refine, num_lvl*num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts, num_refine, num_lvl*num_chosen, -1)
            elif bag_loss_type == 'lvl_mil':
                assert num_refine == 1, "only num_refine == 1 support for now."
                pos_cls_outs = pos_cls_outs.reshape(num_gts, num_refine*num_lvl, num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts, num_refine*num_lvl, num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts, num_refine*num_lvl, num_chosen, -1)
            else:
                raise ValueError()
            # pos_pts = pos_pts.reshape(num_gts, num_refine, num_chosen*num_lvl, -1)
            pos_loss, bag_acc, num_pos = self.bag_loss(pos_cls_outs, pos_ins_outs, pos_valid, gt_labels_all, gt_weights,
                                                       num_lvl, gt_true_bboxes)
            pos_loss = self.loss_cfg['bag_loss_weight'] * pos_loss
            losses.update({"bag_loss": pos_loss, "bag_acc": bag_acc})

        if self.loss_cfg.get("with_neg_loss", True):
            neg_loss = self.loss_cfg["neg_loss_weight"] * self.neg_loss(neg_cls_outs, neg_valid, num_pos)
            losses.update({"neg_loss": neg_loss})

        if self.loss_cfg.get('neg_sampler', 'all') != 'all' and self.loss_cfg.get('neg_sample_rate', -1.0) > 0:
            for key, loss in losses.items():
                if "loss" in key:
                    losses[key] = loss / (1 + self.loss_cfg['neg_sample_rate'])
        return losses


@HEADS.register_module()
class TwoDimMILHead(BaseMILHead):
    def __init__(self, num_classes, in_channel,
                 lvl_feat_pool_weight='ins*cls',

                 loss_cfg=dict(
                     with_bag_loss=True,
                     bag_loss_weight=0.25,
                     refine_bag_policy='only_refine_bag',
                     random_remove_rate=0.4,
                     with_neg_loss=True,
                     neg_loss_type='lvl_mil',
                     neg_loss_weight=0.75,
                     with_gt_loss=True,
                     gt_loss_type="lvl_neg",
                     gt_loss_weight=0.125,
                     mil_loss=dict(),
                 ),
                 debug_cfg=dict(),
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_out',
                         std=0.01,
                         bias_prob=0.01),
                 ), **kwargs):

        self.lvl_feat_pool_weight = lvl_feat_pool_weight
        super(TwoDimMILHead, self).__init__(
            num_classes, loss_cfg=loss_cfg, debug_cfg=debug_cfg, init_cfg=init_cfg, **kwargs
        )
        assert self.sel_share_head_conv, "only support shared head conv for now."

        self.in_channel = in_channel
        self.num_cls_out = num_classes
        # classification branch and selection branch
        chn = self.in_channel
        self.cls_out = nn.Linear(chn, self.num_cls_out)
        self.ins_sel_out = nn.Linear(chn, self.num_cls_out)  # depend on class
        self.lvl_sel_out = nn.Linear(chn, 1)                 # independent over class

    def get_feat_weight(self, cls_prob, ins_prob, gt_labels):
        """
            cls_prob: shape=(n, dk, ...d(i+1), di, num_class)
            gt_labels: shape=(n,)
        """
        if self.lvl_feat_pool_weight == 'ins*cls':
            cls_prob = cls_prob * ins_prob
        elif self.lvl_feat_pool_weight == 'ins':
            cls_prob = ins_prob
        elif self.lvl_feat_pool_weight == 'cls':
            cls_prob = cls_prob
        else:
            raise ValueError
        # feat weight in dim=-2 must sum equal as 1, cause it is a weighted avg pooling in dim=-2
        idx = torch.arange(len(cls_prob)).to(cls_prob.device)
        feat_weight = cls_prob[idx, ..., gt_labels].unsqueeze(dim=-1)  # gt_cls_prob
        feat_weight = feat_weight / feat_weight.sum(dim=-2, keepdim=True)
        return feat_weight

    def get_bag_scores(self, bag_cls_feat, bag_ins_feat, bag_valid, gt_labels):
        """
            feat: shape=(n, num_lvl, num_ins, num_feat_c)
            bag_valid: shape=(n, num_lvl, num_ins, num_class/1)
            gt_labels: shape=(n)
        Return:
            cls_prob: shape=(n, num_lvl, num_ins, num_class)
            ins_prob: shape=(n, num_lvl, num_ins, num_class)
            lvl_prob: shape=(n, num_lvl, 1)
            neg_cls_prob: shape=(num_neg, num_class)
        """
        # 1. get classification score and instance selection score, element-wise mul
        feat_shape = bag_cls_feat.shape
        cls_out = self.cls_out(bag_cls_feat.reshape(-1, feat_shape[-1]))
        cls_out = cls_out.reshape(*bag_cls_feat.shape[:-1], -1)
        cls_prob = self.get_cls_prob(cls_out)

        ins_out = self.ins_sel_out(bag_ins_feat.reshape(-1, feat_shape[-1]))
        ins_out = ins_out.reshape(*bag_ins_feat.shape[:-1], -1)
        ins_prob = ins_out.softmax(dim=-2)
        ins_prob = F.normalize(ins_prob * bag_valid, dim=-2, p=1)    # user should make sure no nan

        # 2. weighted pool feat over instance dim for each level, get lvl score
        feat_weight = self.get_feat_weight(cls_prob, ins_prob, gt_labels)
        feat = (feat_weight * bag_ins_feat).sum(dim=-2)                         # (n, num_lvl, num_feat_C)
        lvl_out = self.lvl_sel_out(feat.reshape(-1, feat_shape[-1]))
        lvl_out = lvl_out.reshape(*feat.shape[:-1], -1)
        lvl_prob = lvl_out.softmax(dim=-2)                                  # (n, num_lvl, 1)
        lvl_prob = lvl_prob * (bag_valid > 0).any(dim=-2).float()
        lvl_prob = F.normalize(lvl_prob, dim=-2)
        return cls_prob, ins_prob, lvl_prob

    def get_neg_scores(self, neg_feat):
        """
            neg_feat: shape=(..., num_feat_c)
        """
        neg_cls_out = self.cls_out(neg_feat.reshape(-1, neg_feat.shape[-1]))
        neg_cls_prob = self.get_cls_prob(neg_cls_out.reshape(*neg_feat.shape[:-1], -1))
        return neg_cls_prob

    def ann_loss(self, ann_cls_prob, lvl_prob, ann_valid, gt_labels, gt_weights):
        """
            ann_cls_prob: (n, num_lvl, num_class)
        """
        n, num_lvl, num_class = ann_cls_prob.shape
        gt_loss_type = self.loss_cfg.get('gt_loss_type', 'lvl_mil')
        if gt_loss_type == 'lvl_mil':
            gt_weights = gt_weights.unsqueeze(dim=-1)
            ann_loss, ann_acc, num_pos = self.mil_loss(ann_cls_prob, lvl_prob, ann_valid, gt_labels, gt_weights)
        elif gt_loss_type == "lvl_pos":
            gt_labels = one_hot(gt_labels, num_class)      # (n, num_class)
            gt_labels = gt_labels.unsqueeze(dim=-2).repeat(1, num_lvl, 1)    # (n, num_lvl, num_class)
            gt_weights = gt_weights.reshape(-1, 1, 1) * ann_valid        # (n, num_lvl, 1/num_class)
            ann_loss = self.base_loss(ann_cls_prob.flatten(0, 1), gt_labels.flatten(0, 1),
                                   gt_weights.flatten(0, 1))  # (n*num_lvl, x)
            num_pos = max((gt_weights > 0).sum(), 1)
            ann_loss = weight_reduce_loss(ann_loss, None, avg_factor=num_pos)
        else:
            raise ValueError
        return ann_loss, num_pos

    def neg_loss(self, neg_cls_prob, lvl_prob, neg_valid, num_pos):
        num_neg, num_class = neg_cls_prob.shape
        neg_labels = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_cls_prob.device)
        neg_loss_type = self.loss_cfg.get('neg_loss_type', 'lvl_neg')
        assert neg_loss_type == 'lvl_neg'
        neg_loss = self.base_loss(neg_cls_prob, neg_labels, neg_valid)
        neg_loss = weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
        # if neg_loss_type == 'lvl_mil':
        #     raise NotImplementedError
        #     # neg_loss = mil_loss(neg_cls_prob, lvl_prob, neg_valid, neg_labels, is_one_hot=True, avg_factor=num_pos)
        # elif neg_loss_type == 'lvl_neg':
        #     # neg_labels = neg_labels.unsqueeze(dim=-2).repeat(1, num_lvl, 1).flatten(0, 1)
        #     neg_loss = self.base_loss(neg_cls_prob, neg_labels, neg_valid)
        #     neg_loss = weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
        # else:
        #     raise ValueError
        return neg_loss

    def bag_loss(self, cls_prob, ins_prob, lvl_prob, bag_valid, gt_labels, gt_weights):
        # 1. weighted to get bag score
        cls_ins_prob = (cls_prob * ins_prob).sum(dim=-2)                         # (n, num_lvl, num_class)
        # prob = (cls_ins_prob * lvl_prob).sum(dim=-2)                             # (n, num_class)
        lvl_valid = (bag_valid > 0).any(dim=-2)
        gt_weights = gt_weights.unsqueeze(dim=-1)
        bag_loss, bag_acc, num_pos = self.mil_loss(cls_ins_prob, lvl_prob, lvl_valid, gt_labels, gt_weights)
        return bag_loss, bag_acc, num_pos

    def get_format_feats(self, bag_data, neg_data):
        for cls_f, ins_f in zip(bag_data.cls_feats, bag_data.ins_feats):
            assert cls_f.shape[1] == 1, "only support num_refine == 1 for now."

        bag_cls_feats, bag_ins_feats, bag_valid = bag_data.cls_feats, bag_data.ins_feats, bag_data.valid
        neg_feats, neg_valid = neg_data.cls_feats, neg_data.valid
        bag_cls_feats = [d.flatten(0, 1) for d in bag_cls_feats]   # remove dim of num_refine
        bag_ins_feats = [d.flatten(0, 1) for d in bag_ins_feats]   # remove dim of num_refine
        bag_valid = [d.flatten(0, 1) for d in bag_valid]           # remove dim of num_refine

        # [num_lvl, (num_gts*num_refine, num_chosen, num_feat_c)]
        # => (num_gts*num_refine, num_lvl, num_chosen, num_feat_c)
        bag_cls_feats, bag_ins_feats, bag_valid = [
            torch.stack(x, dim=1) for x in [bag_cls_feats, bag_ins_feats, bag_valid]]
        return bag_cls_feats, bag_ins_feats, bag_valid, neg_feats, neg_valid

    def loss(self, pos_data: PtAndFeat, neg_data: PtAndFeat, gt_labels, gt_r_points, gt_true_bboxes, gt_weights):
        """
            bag_cls_feat/bag_ins_feat:  shape=(n, num_lvl, num_ins, num_feat_c)
            bag_valid: shape=(n, num_lvl, num_ins, 1/num_class)
            neg_feat:  shape=(num_neg_all_lvl, num_feat_c)
            neg_valid: shape=(num_neg_all_lvl, 1/num_class)
            gt_weights:shape=(n,)
        """
        bag_cls_feat, bag_ins_feat, bag_valid, neg_feat, neg_valid = self.get_format_feats(pos_data, neg_data)
        cls_prob, ins_prob, lvl_prob = self.get_bag_scores(bag_cls_feat, bag_ins_feat, bag_valid, gt_labels)
        neg_feat, neg_valid = [torch.cat(d, dim=0) for d in [neg_feat, neg_valid]]  # [num_neg_all_lvl, num_feat_c]
        neg_cls_prob = self.get_neg_scores(neg_feat)

        losses = {}
        if self.loss_cfg.get("with_ann_loss", True):
            ann_cls_prob, ann_valid = cls_prob[..., -1, :], bag_valid[..., -1, :]  # (n, num_lvl, num_class)
            ann_loss, num_pos = self.ann_loss(ann_cls_prob, lvl_prob, ann_valid.float(), gt_labels, gt_weights)
            losses['ann_loss'] = self.loss_cfg['ann_loss_weight'] * ann_loss

        if self.loss_cfg.get("with_bag_loss", True):
            bag_loss, bag_acc, num_pos = self.bag_loss(cls_prob, ins_prob, lvl_prob, bag_valid, gt_labels, gt_weights)
            bag_loss = self.loss_cfg['bag_loss_weight'] * bag_loss
            losses.update({"bag_loss": bag_loss, "bag_acc": bag_acc})

        if self.loss_cfg.get("with_neg_loss", True):
            neg_loss = self.neg_loss(neg_cls_prob, lvl_prob, neg_valid.float(), num_pos)
            neg_loss = self.loss_cfg["neg_loss_weight"] * neg_loss
            losses.update({"neg_loss": neg_loss})
        return losses

    def estimate(self, feat, gt_labels):
        cls_probs = self.forward(feat, gt_labels)[:-1]
        idx = torch.arange(len(gt_labels)).to(gt_labels.device)
        estimate_res = []
        for i in range(len(self.sel_outs)):
            cls_prob = cls_probs[-1][idx, ..., gt_labels]    # (n, dk)
            best_idx = cls_prob.argmax(dim=-1)               # (n)
            cls_probs = [cls_prob[idx, best_idx] for cls_prob in cls_probs[:-1]]
            estimate_res.append(best_idx)
        estimate_res = torch.stack(estimate_res, dim=-1)
        return estimate_res  # (n, k)

    def predict(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_labels):
        bag_cls_feat, bag_ins_feat, bag_valid, neg_feat, neg_valid = self.get_format_feats(bag_data, grid_data)
        cls_prob, ins_prob, lvl_prob = self.get_bag_scores(bag_cls_feat, bag_ins_feat, bag_valid, gt_labels)
        neg_cls_prob = [self.get_neg_scores(d) for d in neg_feat]
        # to [num_lvl, (num_gt, num_refine=1, num_chosen, X)]
        bag_data.cls_prob = [cls_prob[:, i:i+1] for i in range(cls_prob.shape[1])]
        bag_data.ins_prob = [ins_prob[:, i:i+1] for i in range(ins_prob.shape[1])]
        grid_data.cls_prob = neg_cls_prob
        return bag_data, grid_data


@HEADS.register_module()
class TwoDimMIL2Head(TwoDimMILHead):
    def __init__(self, *args, **kwargs):
        super(TwoDimMIL2Head, self).__init__(*args, **kwargs)

    def ann_loss(self, ann_cls_prob, lvl_prob, ann_valid, gt_labels, gt_weights):
        """
            ann_cls_prob: (n, num_lvl, num_class)
        """
        gt_loss_type = self.loss_cfg.get('gt_loss_type', 'lvl_mil')
        if gt_loss_type == 'lvl_weight':
            gt_weights = gt_weights.unsqueeze(dim=-1)
            ann_loss, ann_acc, num_pos = self.mil_loss(
                ann_cls_prob, lvl_prob.detach(), ann_valid, gt_labels, gt_weights)
            return ann_loss, num_pos
        else:  # lvl_mil, lvl_pos
            return super().ann_loss(ann_cls_prob, lvl_prob, ann_valid, gt_labels, gt_weights)

    def bag_loss(self, cls_prob, ins_prob, lvl_prob, bag_valid, gt_labels, gt_weights):
        # 1. weighted to get bag score
        cls_ins_prob = (cls_prob * ins_prob).sum(dim=-2)                         # (n, num_lvl, num_class)
        # prob = (cls_ins_prob * lvl_prob).sum(dim=-2)                             # (n, num_class)
        lvl_valid = (bag_valid > 0).any(dim=-2)
        gt_weights = gt_weights.unsqueeze(dim=-1)
        bag_loss, bag_acc, num_pos = self.mil_loss(cls_ins_prob, lvl_prob.detach(), lvl_valid, gt_labels, gt_weights)

        a, b = 162, 288
        eps = 1e-7
        num_gt, num_lvl, num_chosen, num_class = cls_prob.shape
        # exp(mean(log(p1, p2, p3))) = (p1*p2*p3) ** (1/3)
        cls_prob1, bag_valid1 = cls_prob[:, :, :a], bag_valid[:, :, :a].float()
        cls_prob2, bag_valid2 = cls_prob[:, :, a:], bag_valid[:, :, a:].float()
        s1 = ((cls_prob1 + eps).log() * bag_valid1).sum(dim=-2) / bag_valid1.sum(dim=-2)
        s2 = ((cls_prob2 + eps).log() * bag_valid2).sum(dim=-2) / bag_valid2.sum(dim=-2)

        s = (s1 - s2).exp()
        s.detach() * lvl_prob
        return bag_loss, bag_acc, num_pos
