
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.core import (multi_apply)
from mmdet.models.builder import HEADS,  build_head
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from mmdet.core.point.cpr_utils.utils import *
from mmdet.core.point.cpr_utils.cpr_refine import PointRefiner
from mmdet.core.point.cpr_utils.sample_feat_extract import SampleFeatExtractor
from mmdet.models.losses.utils import weight_reduce_loss
import os
import copy


class BasePointHead(AnchorFreeHead):
    """
    Base MIL Head
    1. forward_train obtain extra input from dataset (e.g. gt_true_bbox for debug)
    2. got features from FPN
    3. forward with head_conv

    setting:
    1. share conv
    2. number and channel of conv
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 # stacked_convs=4, # number of conv in conv_head
                 sel_share_head_conv=True,  # does selection branch share conv_head with classify branch
                 conv_head_mode='c3333',

                 merge_and_split_fpn_feat=False,
                 merge_before_head_convs=True,

                 normal_cfg=dict(),  # setting for others
                 debug_cfg=dict(open=False),  # setting for debug
                 var_cfg=dict(),     # only for config compatibility of CPRV3HeatHead

                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                 ),
                 **kwargs):
        self.normal_cfg = normal_cfg
        self.debug_cfg = debug_cfg
        self.sel_share_head_conv = sel_share_head_conv
        self.merge_and_split_fpn_feat = merge_and_split_fpn_feat
        self.merge_before_head_convs = merge_before_head_convs
        self.conv_head_mode = conv_head_mode

        # invoke _init_layers in super().__init__
        super().__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.conv_head_mode == 'c3333':
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
        elif self.conv_head_mode == 'c7111':
            chn = self.in_channels
            self.relu = nn.ReLU(inplace=True)
            self.cls_convs = nn.ModuleList()
            # self.ins_convs = nn.ModuleList()
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 7, stride=1, padding=3,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            assert self.sel_share_head_conv
            for i in range(self.stacked_convs):
                self.cls_convs.append(ConvModule(chn, self.feat_channels, 1, stride=1, padding=0,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
                chn = self.feat_channels
        else:
            raise ValueError

        # self.cls_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, 1, 1, 0)
        # self.cls_out = nn.Conv2d(self.point_feat_channels, self.num_cls_out, 1, 1, 0)

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
        """
        Return:
            [num_fpn, (B, C, H, W)]
        """
        if self.merge_and_split_fpn_feat and self.merge_before_head_convs:
            feats = self.merge_and_split_fpn_feats([feats])[0]
        feats_tuples = multi_apply(self.forward_single_with_head_conv, feats)
        if self.merge_and_split_fpn_feat and not self.merge_before_head_convs:
            feats_tuples = self.merge_and_split_fpn_feats(feats_tuples)
        return feats_tuples

    def merge_and_split_fpn_feats(self, feats_tuples):
        """
        feats_tuples: [k, num_lvl, (B, C, H, W)]
        """
        ret_feats = []
        for feats in feats_tuples:
            merge_feat = feats[-1]
            for lvl in range(len(feats)-2, -1, -1):
                merge_feat = feats[lvl] + F.interpolate(merge_feat, size=feats[lvl].shape[2:])
            # merge_feat = merge_feat / len(feats)
            new_feats = [merge_feat]
            for lvl in range(1, len(feats)):
                # cause before downsample using Conv(3, 2, 1), so (12, 21)=>(6, 11)
                new_feats.append(F.avg_pool2d(new_feats[lvl-1], kernel_size=3, stride=2, padding=1))
                # new_feats.append(F.avg_pool2d(feats[lvl-1], kernel_size=2, stride=2, padding=0, ceil_mode=True))
            for f, nf in zip(feats, new_feats):
                assert f.shape == nf.shape, f"{f.shape} vs {nf.shape}"
            feats = new_feats
            ret_feats.append(feats)
        return ret_feats

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

    def get_targets(self):
        pass


@HEADS.register_module()
class CPRV3Head(BasePointHead):
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

                 # 3. mil_head loss and predict score
                 mil_head=dict(),

                 # 4. refiner
                 point_refiner=dict(),

                 normal_cfg=dict(),           # usually for temp setting
                 debug_cfg=dict(open=False),  # setting for debug
                 **kwargs):

        super().__init__(num_classes, in_channels,
                         normal_cfg=normal_cfg, debug_cfg=debug_cfg,
                         **kwargs)

        self.extract_feat_before_fc = extract_feat_before_fc
        s = self.strides
        self.train_pts_extractor = SampleFeatExtractor(**train_pts_extractor,
                                                       strides=s, num_classes=num_classes, debug_cfg=debug_cfg)
        self.refine_pts_extractor = SampleFeatExtractor(**refine_pts_extractor,
                                                        strides=s, num_classes=num_classes, debug_cfg=debug_cfg)
        self.point_refiner = PointRefiner(**point_refiner, debug_cfg=debug_cfg, strides=s,
                                          refine_pts_extractor=self.refine_pts_extractor)

        for key, value in [('num_classes', num_classes), ('sel_share_head_conv', self.sel_share_head_conv)]:
            if key not in mil_head:
                mil_head[key] = value
            else:
                assert value == mil_head[key], f"{value} vs {mil_head[key]}"
        mil_head['debug_cfg'] = debug_cfg
        self.mil_head = build_head(mil_head)

    def extract_feature(self, extractor, cls_feat, ins_feat, input_data, cascade_refine_data):
        if cascade_refine_data is not None:
            input_data.refined_geos = cascade_refine_data.refined_geos
        pos_data, neg_data = extractor(cls_feat, ins_feat, input_data, self.sel_share_head_conv)
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
        losses = self.mil_head.loss(pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes, gt_weights,
                                     img_metas, [f.shape for f in cls_feat])

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
        gt_labels_all = torch.cat(gt_labels, dim=0)
        bag_data, grid_data = self.mil_head.predict(bag_data, grid_data, gt_labels_all)

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


@HEADS.register_module()
class CPRV3HeatHead(CPRV3Head):
    def __init__(self, num_classes, in_channels, *args,
                 var_cfg=dict(
                     with_var_loss=False,
                     var_loss_weight=1.0,
                     loss_type='focal_loss',
                     build_var_layer=True,
                 ),

                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     # override=dict(
                     #     type='Normal',
                     #     name='var_out',
                     #     std=0.01,
                     #     bias_prob=0.01),
                 ),
                 **kwargs):
        self.var_cfg = var_cfg
        if self.var_cfg["with_var_loss"]:
            init_cfg = copy.deepcopy(init_cfg)   # must copy, if the arg is pass to other head, will be changed also
            init_cfg.update(dict(
                override=dict(
                    type='Normal',
                    name='var_out',
                    std=0.01,
                    bias_prob=0.01),
            )),
        super().__init__(num_classes, in_channels, *args, init_cfg=init_cfg, **kwargs)
        self.init_var_layers()

    def init_var_layers(self):
        chn = self.in_channels
        if self.var_cfg['build_var_layer']:
            self.var_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                self.var_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                 conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
                chn = self.feat_channels

        if self.var_cfg["with_var_loss"]:
            self.var_out = nn.Conv2d(chn, self.mil_head.num_cls_out, 3, padding=1)

    def forward(self, feats):
        """
        by default, cascade_cfg['share_feat'] is True;
        only forward of first stage will be called, other stage's forward is not used.
        """
        share_res = self.share_forward(feats)
        return self.unshare_forward(share_res, feats)

    def share_forward(self, feats):
        return super(CPRV3HeatHead, self).forward(feats)

    def unshare_forward(self, share_res, feats):
        if self.var_cfg['build_var_layer']:
            var_feats = []
            for x in feats:
                for var_conv in self.var_convs:
                    x = var_conv(x)
                var_feats.append(x)
        else:
            var_feats = [torch.Tensor([0])] * len(share_res[0])
        return tuple(list(share_res) + [var_feats])

    def loss(self, cls_feat, ins_feat, var_feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, gt_true_bboxes=None,
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
        losses = self.mil_head.loss(pos_data, neg_data, gt_labels_all, gt_r_points, gt_true_bboxes, gt_weights,
                                    img_metas, [f.shape for f in cls_feat])

        if self.debug_cfg.get("print_uncertainty", True):
            self.set_cls_prob(pos_data, neg_data)
            losses["uncertainty"] = self.spatial_centanty(neg_data, gt_labels_all, gt_true_bboxes,
                                                          img_metas, [f.shape for f in cls_feat])
        if self.var_cfg.get("with_var_loss", False):
            # assert len(var_feat[0].shape) == 4, f"{len(var_feat[0].shape)}, var"
            w = self.var_cfg.get("var_loss_weight")
            losses["var_loss"] = w * self.var_loss(var_feat, input_data, cascade_refine_data)
        return losses

    def get_bboxes(self, cls_feat, ins_feat, var_feat, img_metas,
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
        gt_labels_all = torch.cat(gt_labels, dim=0)
        bag_data, grid_data = self.mil_head.predict(bag_data, grid_data, gt_labels_all)

        points, scores, not_refine, ret_cascade_data = self.point_refiner(
            bag_data, grid_data, gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine, cascade_refine_data)

        output_res = self.get_output(img_metas, gt_labels, points, scores, rescale, gt_anns_id,
                                     ret_cascade_data)

        if self.debug_cfg.get("save_masked_img", False):
            self.save_img(grid_data, gt_labels_all, gt_true_bboxes, img_metas, [f.shape for f in cls_feat])
        if cascade_out_fmt:
            return output_res, not_refine, ret_cascade_data

        if with_nms:
            return output_res
        else:
            raise NotImplementedError

    def anchor_points(self, feat, stride):
        b, c, h, w = feat.shape
        device = feat.device
        x, y = torch.arange(w).to(device), torch.arange(h).to(device)
        y, x = torch.meshgrid(y, x)
        pts = torch.stack([x, y], dim=-1) * stride + stride / 2
        return pts

    def var_loss(self, var_feat, input_data, cascade_refine_data):
        var_feat = var_feat[0]
        var_out = self.var_out(var_feat)

        centers = cascade_refine_data.refine_pts
        gt_map = self.generate_gaussian_map(var_out, centers, input_data)
        var_out = var_out.permute(0, 2, 3, 1).flatten(0, -2)
        gt_map = gt_map.permute(0, 2, 3, 1).flatten(0, -2)

        if self.var_cfg.get("loss_type") == "bce":
            loss = F.binary_cross_entropy_with_logits(var_out, gt_map)
            # avg_factor = sum([len(l) for l in input_data.gt_labels])
            loss = weight_reduce_loss(loss, None, avg_factor=None)
        elif self.var_cfg.get("loss_type") == "focal_loss":
            from mmdet.models.point.dense_heads.mil_head import gfocal_loss
            loss = gfocal_loss(var_out.sigmoid(), gt_map)
            avg_factor = sum([len(l) for l in input_data.gt_labels])
            loss = weight_reduce_loss(loss, None, avg_factor=avg_factor)
        else:
            raise ValueError
        return loss

    def generate_gaussian_map(self, var_out, centers, input_data: Input):
        pts = self.anchor_points(var_out, self.strides[0])
        gt_maps = []
        for im_id in range(len(input_data)):
            gt_map_single = self.generate_gaussian_map_single(
                pts, centers[im_id], num_class=var_out.shape[1], input_data_img=input_data.get(im_id))
            gt_maps.append(gt_map_single)
        gt_maps = torch.stack(gt_maps, dim=0)
        return gt_maps

    def generate_gaussian_map_single(self, pts, centers, num_class, input_data_img: Input):
        """ generate gaussian map for each input image.
        for each category:
        1. for each center point, calculate gaussian score of each pt to it.
        2. for each pt, select max gaussian score as it's score.
        return:
            gt_maps: (H, W, num_class)
        """
        # refined_geos = input_data.refined_geos
        # gt_pts = input_data.gt_pts
        assert len(self.strides) == 1
        stride = self.strides[0]

        label2centers = group_by_label(centers, input_data_img.gt_labels)
        gt_maps = torch.zeros(num_class, *pts.shape[:-1], dtype=torch.float32).to(pts.device)
        H, W, _ = pts.shape
        pts = pts.flatten(0, -2)
        for l, centers_l in label2centers.items():
            sigma = stride / 1.5
            dist = ((pts.unsqueeze(dim=0) - centers_l.unsqueeze(dim=1)) / sigma) ** 2  # (num_center, num_pts, 2)
            dist = dist.sum(dim=-1) ** 0.5
            gt_maps[l, ...] = torch.exp(-dist).max(dim=0)[0].reshape(H, W)  # (num_pts,)
        return gt_maps

    def set_cls_prob(self, pos_data, neg_data):
        pos_data.cls_prob = [self.mil_head.get_cls_prob(cls_out) for cls_out in pos_data.cls_outs]
        neg_data.cls_prob = [self.mil_head.get_cls_prob(cls_out) for cls_out in neg_data.cls_outs]

    def spatial_centanty(self, grid_data: PtAndFeat, gt_labels_all, gt_true_bboxes, img_metas, cls_feat_shape, eps=1e-8):
        assert len(grid_data.cls_prob) == 1, "only support single FPN level"
        B, C, H, W = cls_feat_shape[0]
        gt_labels, start_i = [], 0
        for gt_true_bboxes_img in gt_true_bboxes:
            end_i = start_i + len(gt_true_bboxes_img)
            gt_labels.append(gt_labels_all[start_i:end_i])
            start_i = end_i

        all_std_scores = []
        for img_i, data in enumerate(grid_data.split_each_img()):
            pts = data.pts[0].reshape(-1, 3)[:, :2]
            score_maps = data.cls_prob[0].flatten(0, -2)

            for gt_i, (l, bbox) in enumerate(zip(gt_labels[img_i], gt_true_bboxes[img_i])):
                idx = (pts[:, 0] < bbox[2]) & (pts[:, 0] > bbox[0]) & (pts[:, 1] < bbox[3]) & (pts[:, 1] > bbox[1])
                scores = score_maps[:, l][idx]
                # pts[idx] -
                if len(scores) > 0:
                    scores /= scores.sum()
                    v = -(scores * (scores+eps).log()).sum()  # v = scores.std()
                    all_std_scores.append(v)

            # # visualize debug
            # img_meta = img_metas[img_i]
            # import matplotlib.pyplot as plt
            # from mmdet.core.point.cpr_utils.test_cpr import TestCPRHead
            # pad_img = TestCPRHead.load_pad_img(img_meta)
            # if img_i == 0:
            #     plt.imshow(pad_img)
            #     TestCPRHead.show_bbox(gt_true_bboxes[img_i].detach().cpu().numpy())
            #     plt.show()
            # for gt_i, (l, bbox) in enumerate(zip(gt_labels[img_i], gt_true_bboxes[img_i])):
            #     if img_i == 0 and gt_i == 0:
            #         plt.imshow(score_maps[:, l].reshape(H, W).detach().cpu().numpy())
            #         plt.show()
            # assert len(pts) == len(score_maps)
        uncertainty = torch.stack(all_std_scores, dim=0).mean() if len(all_std_scores) > 0 \
            else torch.zeros(1).to(gt_labels_all.device)
        return uncertainty

    def save_img(self, grid_data, gt_labels_all, gt_true_bboxes, img_metas, cls_feat_shape):
        assert len(grid_data.cls_prob) == 1, "only support single FPN level"
        B, C, H, W = cls_feat_shape[0]
        stride = self.strides[0]
        gt_labels, start_i = [], 0
        for gt_true_bboxes_img in gt_true_bboxes:
            end_i = start_i + len(gt_true_bboxes_img)
            gt_labels.append(gt_labels_all[start_i:end_i])
            start_i = end_i

        for img_i, data in enumerate(grid_data.split_each_img()):
            pts = data.pts[0].reshape(-1, 3)[:, :2]
            score_maps = data.cls_prob[0].flatten(0, -2)

            # visualize debug
            img_meta = img_metas[img_i]
            # if img_meta['filename'] == 'data/coco/train2017/000000041950.jpg':
            if img_i == 0:
                import matplotlib.pyplot as plt
                from mmdet.core.point.cpr_utils.test_cpr import TestCPRHead
                from ssdcv.vis.plt_paper_config import get_clear_fig
                pad_img = TestCPRHead.load_pad_img(img_meta, (H*stride, W*stride))
                label2bboxes = group_by_label(gt_true_bboxes[img_i], gt_labels[img_i])
                for l, bboxes in label2bboxes.items():
                    mask = score_maps[:, l].reshape(H, W).detach().cpu().numpy()
                    mask = (mask - mask.min()) / (mask.max() - mask.min())
                    mask_img = TestCPRHead.mask_img(mask, pad_img)

                    fig, ax = get_clear_fig(H*stride, W*stride)
                    plt.imshow(mask_img)
                    TestCPRHead.show_bbox(bboxes.detach().cpu().numpy())
                    # plt.show()

                    plt.savefig(get_save_path(f"exp/debug/CPRV3HeatHead/", img_meta['filename'], l, self,
                                              self.debug_cfg.get("save_masked_img_epoch", -1)))
                    plt.clf()
                SaveTime[0] += 1
                if SaveTime[0] >= self.debug_cfg.get("save_masked_img_time", 1):
                    os.system("python exp/tools/killgpu.py 0-7")
                break


StageId = dict()
SaveTime = [0]
def get_save_path(save_dir, filepath, l, cpr_head, epoch_id):
    if cpr_head not in StageId:
        StageId[cpr_head] = len(StageId)
    filename = os.path.split(filepath)[-1]
    filename = os.path.splitext(filename)[0]
    import glob
    file_prefix = f"{save_dir}/{filename}_c{str(l).zfill(2)}_s{str(StageId[cpr_head])}_e{str(epoch_id).zfill(2)}_"
    files = glob.glob(f"{file_prefix}*.jpg")
    save_path = f"{file_prefix}{str(len(files)).zfill(2)}.jpg"
    return save_path


if __name__ == '__main__':

    print()
