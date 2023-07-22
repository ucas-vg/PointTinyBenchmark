
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.core import (multi_apply)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from mmdet.core.point.cpr_utils.utils import *
from mmdet.core.point.cpr_utils.cpr_refine import PointRefiner
from mmdet.core.point.cpr_utils.sample_feat_extract import SampleFeatExtractor


class BaseCPRHead(AnchorFreeHead):
    """
    Base MIL Head
    1. forward_train obtain extra input from dataset (e.g. gt_true_bbox for debug)
    2. got features from FPN
    3. forward with head_conv
    4. forward with head_fc

    setting:
    1. share conv, fc, out
    2. number and channel of conv, fc
    """
    def __init__(self,
                 num_classes,
                 in_channels,

                 # head_feat_cfg
                 # stacked_convs=4, # number of conv in conv_head
                 stacked_fcs=0,             # number of fc in fc_head
                 fc_out_channels=1024,      # channel of fc in fc_head
                 sel_share_head_conv=True,  # does selection branch share conv_head with classify branch
                 sel_share_head_fc=True,
                 sel_share_head_out=False,  # does selection branch share logit with classify branch
                 out_bg_cls=False,          # does last branch need out logit for background
                 prob_cls_type='sigmoid',   # 'sigmoid', 'softmax', 'normed_sigmoid'
                 normed_sigmoid_p=1,        # p for prob_cls_type=='normed_sigmoid'

                 loss_mil=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=1.0),
                 normal_cfg=dict(),  # setting for others
                 loss_cfg=dict(),    # setting for loss
                 debug_cfg=dict(open=False),  # setting for debug

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
                 **kwargs):
        # head_feat_cfg
        self.num_cls_out = num_classes + 1 if out_bg_cls else num_classes
        self.stacked_fcs = stacked_fcs
        self.fc_out_channels = fc_out_channels
        self.sel_share_head_conv = sel_share_head_conv
        self.sel_share_head_fc = sel_share_head_fc
        self.sel_share_head_out = sel_share_head_out
        self.prob_cls_type = prob_cls_type
        self.normed_sigmoid_p = normed_sigmoid_p

        self.binary_ins = loss_mil['binary_ins']
        self.loss_cfg = loss_cfg
        self.normal_cfg = normal_cfg
        self.debug_cfg = debug_cfg

        # invoke _init_layers in super().__init__
        super().__init__(num_classes, in_channels, loss_cls=loss_mil, init_cfg=init_cfg, **kwargs)

        self.loss_mil = build_loss(loss_mil)

    def _init_layers(self):
        """Initialize layers of the head."""
        chn = self.in_channels
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.ins_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            if not self.sel_share_head_conv:
                self.ins_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            chn = self.feat_channels

        # self.cls_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, 1, 1, 0)
        # self.cls_out = nn.Conv2d(self.point_feat_channels, self.num_cls_out, 1, 1, 0)

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

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None,
                      gt_true_bboxes=None, proposal_cfg=None, **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_true_bboxes=gt_true_bboxes)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        return self.forward_with_head_conv(feats)

    def forward_with_head_conv(self, feats):
        """
        Return:
            [num_fpn, (B, C, H, W)]
        """
        return multi_apply(self.forward_single_with_head_conv, feats)

    def forward_single_with_head_conv(self, x):
        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        if not self.sel_share_head_conv:
            ins_feat = x
            for ins_conv in self.ins_convs:
                ins_feat = ins_conv(ins_feat)
        else:
            ins_feat = cls_feat
        return cls_feat, ins_feat

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

    def get_targets(self):
        pass

    # def get_bboxes(self):
    #     pass
    #
    # def loss(self):
    #     pass


@HEADS.register_module()
class CPRV2Head(BaseCPRHead):
    def __init__(self, num_classes, in_channels,

                 # 2. feature extractor
                 extract_feat_before_fc=True,
                 train_pts_extractor=dict(
                     pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                     neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3),
                 ),
                 refine_pts_extractor=dict(
                     pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
                     neg_generator=dict(type='AnchorPtFeatGenerator', scale_factor=1.0),
                 ),

                 # 4. refiner
                 point_refiner=dict(),

                 # 3. loss
                 loss_cfg=dict(
                     loss_type=0,
                     with_neg=True,
                     neg_loss_weight=0.75,
                     refine_bag_policy='independent_with_gt_bag',
                     random_remove_rate=0.4,
                     with_gt_loss=False,
                     gt_loss_weight=0.125,
                     with_mil_loss=True,
                 ),

                 normal_cfg=dict(),           # usually for temp setting
                 debug_cfg=dict(open=False),  # setting for debug
                 **kwargs):
        super().__init__(num_classes, in_channels,
                         normal_cfg=normal_cfg, loss_cfg=loss_cfg, debug_cfg=debug_cfg,
                         **kwargs)
        self.extract_feat_before_fc = extract_feat_before_fc
        s = self.strides
        self.train_pts_extractor = SampleFeatExtractor(**train_pts_extractor,
                                                       strides=s, num_classes=num_classes, debug_cfg=debug_cfg)
        self.refine_pts_extractor = SampleFeatExtractor(**refine_pts_extractor,
                                                        strides=s, num_classes=num_classes, debug_cfg=debug_cfg)
        self.point_refiner = PointRefiner(**point_refiner, debug_cfg=debug_cfg, strides=s,
                                          refine_pts_extractor=self.refine_pts_extractor)

    def extract_feature(self, extractor, cls_feat, ins_feat, input_data, cascade_refine_data):
        if cascade_refine_data is not None:
            input_data.refined_geos = cascade_refine_data.refined_geos
        pos_data, neg_data = extractor(cls_feat, ins_feat, input_data, self.sel_share_head_conv)
        pos_data.cls_outs, pos_data.ins_outs = self.forward_with_head_fc_and_out(pos_data.cls_feats, pos_data.ins_feats)
        neg_data.cls_outs = self.forward_with_head_fc_and_out(neg_data.cls_feats)
        return pos_data, neg_data

    def loss(self, cls_feat, ins_feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, gt_true_bboxes=None,
             gt_weights=None, cascade_refine_data=None):
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None
        gt_r_points = [pts.reshape(len(labels), -1, *pts.shape[1:]) for pts, labels in zip(gt_points, gt_labels)]
        # gt_r_points = [pts.unsqueeze(dim=1) for pts in gt_points]

        input_data = Input(img_metas, gt_r_points, gt_points, gt_labels, gt_points_ignore, gt_true_bboxes, gt_weights)
        pos_data, neg_data = self.extract_feature(self.train_pts_extractor, cls_feat, ins_feat,
                                                  input_data, cascade_refine_data)

        gt_labels_all = torch.cat(gt_labels, dim=0)
        gt_weights = [torch.FloatTensor([1.0] * len(l)) for l in gt_labels] if gt_weights is None else gt_weights
        gt_weights = torch.cat(gt_weights, dim=0).to(gt_labels_all.device)
        return getattr(self, f'loss{self.loss_cfg["loss_type"]}')(pos_data, neg_data, gt_labels_all, gt_r_points,
                                                                  gt_true_bboxes, gt_weights)

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

    def loss0(self, pos_data: PtAndFeat, neg_data: PtAndFeat, gt_labels_all, gt_r_points,
              gt_true_bboxes=None, gt_weights=None):
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
        from mmdet.models.losses.utils import weight_reduce_loss
        pos_cls_outs, pos_ins_outs, neg_cls_outs = pos_data.cls_outs, pos_data.ins_outs, neg_data.cls_outs
        pos_pts, neg_pts = pos_data.pts, neg_data.pts
        pos_valid, neg_valid = pos_data.valid, neg_data.valid
        assert len(pos_cls_outs) == 1, f"{len(pos_cls_outs)}"
        pos_cls_outs, pos_ins_outs, pos_pts = pos_cls_outs[0], pos_ins_outs[0], pos_pts[0]
        pos_valid = pos_valid[0]
        neg_cls_outs, neg_pts = torch.cat(neg_cls_outs, dim=0), torch.cat(neg_pts, dim=0)
        neg_valid = torch.cat(neg_valid, dim=0)

        losses = {}
        num_gts, num_refine, num_chosen, _ = pos_pts.shape
        if self.loss_cfg.get('with_gt_loss', False):
            gt_cls_outs = pos_cls_outs[..., -1, :].reshape(num_gts * num_refine, -1)
            gt_cls_prob = self.get_cls_prob(gt_cls_outs)

            gt_loss_type = self.loss_cfg.get('gt_loss_type', 'gt_refine')
            if gt_loss_type == 'mil':
                raise NotImplementedError
            else:
                if gt_loss_type == 'gt_refine':
                    gt_labels_rep = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_valid = pos_valid[..., -1, :].reshape(num_gts * num_refine, -1)
                    gt_weights_rep = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                    gt_weights_rep = gt_valid.float() * gt_weights_rep.reshape(-1, 1)
                elif gt_loss_type == 'gt':
                    gt_labels_rep = gt_labels_all
                    gt_weights_rep = pos_valid[:, 0, -1, :].float() * gt_weights.reshape(-1, 1)
                    gt_cls_prob = gt_cls_prob.reshape(num_gts, num_refine, -1)[:, 0]
                else:
                    raise ValueError()
                gt_labels = torch.full(gt_cls_prob.shape, 0., dtype=torch.float32).to(gt_cls_prob.device)
                gt_labels[torch.arange(len(gt_labels)), gt_labels_rep] = 1
                num_pos = max((gt_weights_rep > 0).sum(), 1)

                gt_loss = self.loss_mil.gfocal_loss(gt_cls_prob, gt_labels, gt_weights_rep)
                gt_loss = self.loss_cfg['gt_loss_weight'] * weight_reduce_loss(gt_loss, None, avg_factor=num_pos)
            losses['gt_loss'] = gt_loss

        if self.loss_cfg.get('with_mil_loss', True):
            refine_bag_policy = self.loss_cfg["refine_bag_policy"]
            if refine_bag_policy == 'independent_with_gt_bag':
                # treat num_refine as independent bags
                pos_pts = pos_pts.reshape(num_gts * num_refine, num_chosen, 3)
                pos_cls_outs = pos_cls_outs.reshape(num_gts * num_refine, num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts * num_refine, num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts * num_refine, num_chosen, -1)
                pos_weights = gt_weights.unsqueeze(dim=1).repeat(1, num_refine).flatten()
                gt_labels_all = gt_labels_all.unsqueeze(dim=1).repeat(1, num_refine).flatten()
            elif refine_bag_policy == 'merge_to_gt_bag':
                pos_pts = pos_pts.reshape(num_gts, num_refine * num_chosen, 3)
                pos_cls_outs = pos_cls_outs.reshape(num_gts, num_refine * num_chosen, -1)
                pos_ins_outs = pos_ins_outs.reshape(num_gts, num_refine * num_chosen, -1)
                pos_valid = pos_valid.reshape(num_gts, num_refine * num_chosen, -1)
                pos_weights = gt_weights
            elif refine_bag_policy == 'only_refine_bag':
                si = 1 if num_refine > 1 else 0
                pos_pts = pos_pts[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, 3)
                pos_cls_outs = pos_cls_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_ins_outs = pos_ins_outs[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_valid = pos_valid[:, si:].reshape(num_gts, (num_refine - si) * num_chosen, -1)
                pos_weights = gt_weights
            else:
                raise ValueError
            pos_weights = pos_valid.float() * pos_weights.reshape(-1, 1, 1)

            self.random_remove(pos_pts, neg_pts, random_remove_rate=self.loss_cfg['random_remove_rate'])

            pos_cls_prob = self.get_cls_prob(pos_cls_outs)
            pos_loss, bag_acc, num_pos = self.loss_mil(pos_cls_prob, pos_ins_outs, gt_labels_all, pos_weights)
            losses.update({"pos_loss": pos_loss, "bag_acc": bag_acc})

        if self.loss_cfg.get("with_neg", True):
            neg_prob = self.get_cls_prob(neg_cls_outs)
            num_neg, num_class = neg_prob.shape
            neg_labels = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_prob.device)
            loss_weights = self.loss_cfg["neg_loss_weight"]
            neg_valid = neg_valid.reshape(num_neg, -1)

            neg_loss = self.loss_mil.gfocal_loss(neg_prob, neg_labels, neg_valid.float())
            neg_loss = loss_weights * weight_reduce_loss(neg_loss, None, avg_factor=num_pos)
            losses.update({"neg_loss": neg_loss})
        return losses

    def get_bboxes(self, cls_feat, ins_feat, img_metas,
                   cfg=None, rescale=False, with_nms=True,
                   gt_bboxes=None, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None, gt_anns_id=None,
                   not_refine=None, cascade_out_fmt=False, cascade_refine_data=None):
        """
        Args:
            gt_bboxes: [num_img, (num_gts*num_refine, 4)]
            gt_labels: [num_img, (num_gts,)]
            gt_bboxes_ignore:
            gt_true_bboxes: [num_img, (num_gts, 4)] or None
        """
        assert len(gt_labels) > 0
        gt_points = self.pseudo_bbox_to_center(gt_bboxes)
        gt_points_ignore = self.pseudo_bbox_to_center(gt_bboxes_ignore) if gt_bboxes_ignore else None
        gt_r_points = [pts.reshape(len(labels), -1, *pts.shape[1:]) for pts, labels in zip(gt_points, gt_labels)]
        # gt_r_points = [pts.unsqueeze(dim=1) for pts in gt_points]

        input_data = Input(img_metas, gt_r_points, gt_points, gt_labels, gt_points_ignore, gt_true_bboxes)
        bag_data, grid_data = self.extract_feature(self.refine_pts_extractor, cls_feat, ins_feat,
                                                   input_data, cascade_refine_data)
        bag_data.cls_prob = [self.get_cls_prob(bag_cls_feat_lvl) for bag_cls_feat_lvl in bag_data.cls_outs]
        grid_data.cls_prob = [self.get_cls_prob(grid_cls_outs_lvl) for grid_cls_outs_lvl in grid_data.cls_outs]

        points, scores, not_refine, ret_cascade_data = self.point_refiner(
            bag_data, grid_data, gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine, cascade_refine_data)

        output_res = self.get_output(img_metas, gt_labels, points, scores, rescale, gt_anns_id,
                                     ret_cascade_data)
        if cascade_out_fmt:
            return output_res, not_refine, ret_cascade_data

        if with_nms:
            return output_res
        else:
            raise NotImplementedError

    def get_output(self, img_metas, gt_labels, points, scores, rescale, gt_anns_id, ret_cascade_data):
        """result that ready write to json file"""
        assert sum([len(l) for l in gt_labels]) == sum([len(p) for p in points])
        final_det_bboxes = []
        det_bboxes = self.center_to_pseudo_bbox(points)
        for im_id, bboxes in enumerate(det_bboxes):
            if rescale:
                scale_factor = img_metas[im_id]['scale_factor']
                bboxes /= bboxes.new_tensor(scale_factor)
            scores_img = scores[im_id].unsqueeze(dim=-1)
            anns_id = gt_anns_id[im_id].unsqueeze(dim=-1).type_as(scores_img)
            if self.norm_cfg.get('out_geo', False):
                geos_im = ret_cascade_data.refined_geos[im_id]
                if rescale:
                    scale_factor = img_metas[im_id]['scale_factor']
                    geos_im = self.scale_geos(geos_im, scale_factor)  # [num_gt]
                geos_im = fill_list_to_tensor(geos_im)
                geos_im = geos_im.reshape(len(geos_im), -1)
                final_det_bboxes.append(torch.cat((bboxes, scores_img, anns_id, geos_im), dim=-1))
            else:
                final_det_bboxes.append(torch.cat((bboxes, scores_img, anns_id), dim=-1))
        return list(zip(final_det_bboxes, gt_labels))

    def get_targets(self):
        pass

    def scale_geos(self, geos_im, scale_factor):
        for gt_id, geos_gt in enumerate(geos_im):
            geos_gt[:] /= geos_gt.new_tensor(scale_factor[:2])
        return geos_im

    def pseudo_bbox_to_center(self, gt_bboxes):
        """
        Transform pseudo bbox to center point
        Args:
            gt_bboxes: [num_imgs, (num_pts, 2)]
        Returns:
        """
        return [(gt_bboxes_img[:, :2] + gt_bboxes_img[:, 2:]) / 2
                for gt_bboxes_img in gt_bboxes]

    def center_to_pseudo_bbox(self, centers, pseudo_wh=(16, 16)):
        """
        Returns:
        """
        if not isinstance(pseudo_wh, torch.Tensor):
            pseudo_wh = centers[0].new_tensor(pseudo_wh)
        return [torch.cat([center - pseudo_wh / 2, center + pseudo_wh / 2], dim=-1) for center in centers]


if __name__ == '__main__':
    m = BaseMILHead(80, 256, stacked_fcs=1)
    feats = [torch.rand(2, 256, 32, 32), torch.rand(2, 256, 16, 16)]
    print(m)
    m.forward_with_head_conv(feats)
    feats = [f.permute(0, 2, 3, 1) for f in feats]
    m.forward_with_head_fc_and_out(feats)
    print()
