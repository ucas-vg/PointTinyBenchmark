import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .cascade_roi_head import CascadeRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class EP2RBHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, num_stages, bbox_head, **kwargs):
        super(EP2RBHead, self).__init__(bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk = 7

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.cdb = build_head(dict(type='ConvConcreteDB', cfg=None, planes=256))
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      stage,
                      x,
                      img_metas,
                      pseudo_boxes,
                      proposal_list,
                      neg_proposal_list,
                      neg_weight_list,
                      gt_bboxes,
                      gt_labels,
                      dynamic_weight,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ):

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            if stage == 'with_rpn':
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                bbox_results = self._bboxdet_forward_train(x, sampling_results,
                                                           gt_bboxes, gt_labels, dynamic_weight,  # add by fei
                                                           img_metas)
                losses.update(bbox_results['loss_bbox'])
                return losses
            else:
                bbox_results = self._bbox_forward_train(x, pseudo_boxes, proposal_list, neg_proposal_list,
                                                        neg_weight_list,
                                                        gt_bboxes, gt_labels, dynamic_weight,
                                                        img_metas, stage)

                losses.update(bbox_results['loss_instance_mil'])
                return losses, bbox_results['pseudo_boxes'], bbox_results['dynamic_weight']

    def _bboxdet_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, ann_weight,
                               img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois, gt_points=None, stage=1, )

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, ann_weight, self.train_cfg)  ## add by fei
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train(self, x, proposal_list_base, proposals_list, neg_proposal_list, neg_weight_list, gt_points,
                            gt_labels,
                            cascade_weight,
                            img_metas, stage):
        """Run forward function and calculate loss for box head in training."""

        # self.show_box(aug_proposals_list,neg_proposal_list,neg_weight_list, gt_points,img_metas)

        rois = bbox2roi(proposals_list)
        bbox_results = self._bbox_forward(x, rois, gt_points, stage)
        num_instance = bbox_results['num_instance']
        gt_labels = torch.cat(gt_labels)

        # loss = F.nll_loss(log_soft_out,gt_labels)
        reg_box = bbox_results['bbox_pred']
        if neg_proposal_list is not None:
            neg_rois = bbox2roi(neg_proposal_list)
            neg_bbox_results = self._bbox_forward(x, neg_rois, None, stage)
            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
        else:
            neg_cls_scores = None
            neg_weights = None
        boxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 4),
                                                      reg_box.reshape(-1, 4)).reshape(reg_box.shape)

        ######################################################################################################333
        # if stage < self.num_stages - 1:
        #     proposals_list_to_merge = proposals_list
        # else:
        #     proposals_list_to_merge = boxes_pred
        proposals_list_to_merge = proposals_list

        pseudo_boxes, mean_ious, boxes2, dynamic_weight = self.merge_box(bbox_results, proposals_list_to_merge,
                                                                         gt_labels,
                                                                         gt_points,
                                                                         img_metas, stage)
        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=[i.sum(dim=-1) for i in dynamic_weight])
        # pseudo_boxes1, mean_ious1, boxes2,dynamic_weight1 = self.merge_box(bbox_results, boxes_pred, gt_labels, gt_points,
        #                                                             img_metas)
        # self.show_box(pseudo_boxes,, neg_proposal_list, neg_weight_list, gt_points, img_metas)
        pseudo_boxes = torch.cat(pseudo_boxes)
        if stage == self.num_stages - 1:
            retrain_weights = self._bac_assigner(torch.cat(proposals_list), torch.cat(proposal_list_base))
        else:
            retrain_weights = None

        gt_boxes_target = self.bbox_head.bbox_coder.encode(
            torch.cat(proposals_list).reshape(-1, 4),
            pseudo_boxes.detach().unsqueeze(1).expand(reg_box.shape).reshape(-1, 4))

        loss_instance_mil = self.bbox_head.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
                                                neg_cls_scores, neg_weights,
                                                boxes_pred, gt_labels,
                                                torch.cat(proposal_list_base), label_weights=cascade_weight,
                                                retrain_weights=retrain_weights)
        loss_instance_mil.update({"mean_ious": mean_ious})

        bbox_results.update(loss_instance_mil=loss_instance_mil)

        return bbox_results

    def _bbox_forward(self, x, rois, gt_points=None, stage=1):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # # if stage == 2:
        # bbox_feats = self.cdb(bbox_feats)

        cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)

        # positive sample
        if gt_points is not None:
            num_gt = torch.cat(gt_points).shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_points}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])
            # mil_score = F.softmax(cls_score, dim=-1) * F.softmax(posi_score, dim=-2)
            # instance_mil_score = mil_score.sum(dim=1, keepdim=False)
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=num_gt)
            return bbox_results
        # megative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=None)
            return bbox_results

    def forward_train_mask(self, x, img_metas,
                           proposals_list,
                           gt_true_bboxes, gt_labels,
                           dynamic_weight,
                           gt_bboxes_ignore, gt_masks=None
                           ):
        losses = dict()
        # bbox head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, proposals_list,
                                                    gt_true_bboxes, gt_labels, dynamic_weight,
                                                    img_metas)

            losses.update(mask_results['loss_mask'])
        return losses
        # , mask_results['pseudo_boxes'], mask_results['dynamic_weight']

    def _mask_forward_train(self, x, proposals_list, gt_bboxes, gt_labels, dynamic_weight,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""

        rois = bbox2roi(proposals_list)

        mask_results = self._mask_forward(x, rois)

        loss_masks = self.mask_head.loss(mask_results['mask_cls_pred'], mask_results['mask_ins_pred'],
                                         torch.cat(gt_labels), dynamic_weight)
        # from exp.tools.visual import show_imgs
        # show_imgs(mask_results['mask_cls_pred'], img_metas, gt_labels, proposals_list, gt_bboxes)

        mask_results.update(loss_mask=loss_masks)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois, roi_scale_factor=1.0)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_cls_pred, mask_ins_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_cls_pred=mask_cls_pred, mask_ins_pred=mask_ins_pred, mask_feats=mask_feats)
        return mask_results

    def merge_box_single(self, cls_score, dynamic_weight, gt_point, gt_label, aug_proposals, img_metas, stage):
        if stage < self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'
        elif stage == self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'

        aug_proposals = aug_proposals.reshape(cls_score.shape[0], cls_score.shape[1], 4)
        h, w, c = img_metas['img_shape']
        num_gt, num_gen = aug_proposals.shape[:2]
        # aug_proposals = aug_proposals.reshape(-1,4)
        cls_score = cls_score[torch.arange(num_gt), :, gt_label]

        # ##vote
        if merge_mode == 'vote':
            weight = cls_score.clamp(0.3, 1).unsqueeze(2).repeat([1, 1, 4])
            weight = weight.softmax(dim=1)
            boxes1 = (aug_proposals * weight).sum(dim=1)

            grid_box = aug_proposals.round().long()
            grid_box[:, :, 0::2] = grid_box[:, :, 0::2].clamp(0, w)
            grid_box[:, :, 1::2] = grid_box[:, :, 1::2].clamp(0, h)
            mask = cls_score.new_full((num_gt, w, h), 0)
            boxes = []
            for i in range(num_gt):
                for j, b in enumerate(grid_box[i]):
                    mask[i, b[0]:b[2], b[1]:b[3]] += cls_score[i, j]
                id = torch.nonzero(mask[i] / torch.max(mask[i]) > self.threshold)
                if id.max() == 0:
                    torch.nonzero(mask[i] / torch.max(mask[i]) > 0)
                max_X = max(id[:, 0])
                max_Y = max(id[:, 1])
                min_X = min(id[:, 0])
                min_Y = min(id[:, 1])
                boxes.append(torch.tensor([min_X, min_Y, max_X, max_Y]))
            boxes = torch.stack(boxes).to(mask.device)
            return boxes1, boxes, mask

        ##weighted
        if merge_mode == 'weighted_cls':
            # dynamic_weight = (dynamic_weight > 0.01) * dynamic_weight
            # if dynamic_weight.max() ==0:
            #     dynamic_weight
            weight = cls_score.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (aug_proposals * weight).sum(dim=1)
            # print(weight.sum(dim=1))
            # print(boxes)
            return boxes, None, None
        if merge_mode == 'weighted_cls_topk':
            cls_score_, idx = cls_score.topk(k=7, dim=1)
            weight = cls_score_.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (aug_proposals[torch.arange(aug_proposals.shape[0]).unsqueeze(1), idx] * weight).sum(dim=1)
            # print(weight.sum(dim=1))
            # print(boxes)
            return boxes, None, None

        if merge_mode == 'weighted_clsins':
            # dynamic_weight = (dynamic_weight > 0.01) * dynamic_weight
            # if dynamic_weight.max() ==0:
            #     dynamic_weight
            weight = dynamic_weight.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (aug_proposals * weight).sum(dim=1)
            # print(weight.sum(dim=1))
            # print(boxes)
            return boxes, None, None

        if merge_mode == 'weighted_clsins_topk':
            dynamic_weight_, idx = dynamic_weight.topk(k=self.topk, dim=1)
            weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (aug_proposals[torch.arange(aug_proposals.shape[0]).unsqueeze(1), idx] * weight).sum(dim=1)
            # print(weight.sum(dim=1))
            # print(boxes)
            return boxes, None, None

        if merge_mode == 'max_score':
            _, idx = cls_score.max(dim=1)
            pseudo_box = aug_proposals[torch.arange(aug_proposals.shape[0]), idx]

            return pseudo_box, None, None
        if merge_mode == 'max_clsins':
            _, idx = dynamic_weight.max(dim=1)
            pseudo_box = aug_proposals[torch.arange(aug_proposals.shape[0]), idx]

            return pseudo_box, None, None
            # 
            # grid_box = aug_proposals.round().long()
            # grid_box[:, :, 0::2] = grid_box[:, :, 0::2].clamp(0, w)
            # grid_box[:, :, 1::2] = grid_box[:, :, 1::2].clamp(0, h)
            # mask = cls_score.new_full((num_gt, w, h), 0)
            # boxes = []
            # for i in range(num_gt):
            #     for j, b in enumerate(grid_box[i]):
            #         mask[i, b[0]:b[2], b[1]:b[3]] += dynamic_weight[i, j]
            #     id = torch.nonzero(mask[i]  > self.threshold)
            #     if id.shape[0]==0:
            #         id=torch.nonzero(mask[i] >= 0)
            #     max_X = max(id[:, 0])
            #     max_Y = max(id[:, 1])
            #     min_X = min(id[:, 0])
            #     min_Y = min(id[:, 1])
            #     boxes.append(torch.tensor([min_X, min_Y, max_X, max_Y]))
            # boxes = torch.stack(boxes).to(mask.device)
            # return boxes1, boxes, mask

    def merge_box(self, bbox_results, aug_proposals_list, gt_labels, gt_points, img_metas, stage):
        cls_scores = bbox_results['cls_score']
        ins_scores = bbox_results['ins_score']
        num_instances = bbox_results['num_instance']
        # num_gt = len(gt_labels)

        if stage == 0:
            cls_scores = cls_scores.softmax(dim=-1)
        else:
            cls_scores = cls_scores.sigmoid()
        ins_scores = ins_scores.softmax(dim=-2)
        dynamic_weight = (cls_scores * ins_scores).detach()
        dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
        # split batch
        batch_gt = [len(b) for b in gt_points]
        cls_scores = torch.split(cls_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
        if not isinstance(aug_proposals_list, list):
            aug_proposals_list = torch.split(aug_proposals_list, batch_gt)
        stage_ = [stage for _ in range(len(cls_scores))]
        boxes1, boxes, mask = multi_apply(self.merge_box_single, cls_scores, dynamic_weight_list, gt_points, gt_labels,
                                          aug_proposals_list,
                                          img_metas, stage_)
        # pseudo_boxes = torch.cat(boxes)
        # # mean_ious =torch.tensor(mean_ious).to(gt_point.device)

        pseudo_boxes1 = torch.cat(boxes1).detach()
        # mean_ious =torch.tensor(mean_ious).to(gt_point.device)
        iou1 = bbox_overlaps(pseudo_boxes1, torch.cat(gt_points), is_aligned=True)
        if self.test_mean_iou and stage == 1:
            self.sum_iou += iou1.sum()
            self.sum_num += len(iou1)
            print(self.sum_iou / self.sum_num)
        mean_ious1 = iou1.mean()
        pseudo_boxes1 = torch.split(pseudo_boxes1, batch_gt)
        return list(pseudo_boxes1), mean_ious1, boxes, dynamic_weight_list

    def _bac_assigner(self, bbox, gt_bbox):
        num_gt = gt_bbox.shape[0]
        bbox = bbox.reshape(num_gt, -1, 4)
        gt_bbox = gt_bbox.unsqueeze(1).expand(bbox.shape)
        iou = bbox_overlaps(bbox, gt_bbox, is_aligned=True)
        box_weight = (iou > 0.5).long()
        _, idx = iou.max(dim=1)
        box_weight[idx] = 1
        return box_weight

    def show_box(self, aug_proposals_list, neg_proposal_list, neg_weight_list, gt_points, img_metas):
        import cv2
        import numpy as np
        for i in range(len(img_metas)):
            pos_box = aug_proposals_list[i]
            neg_box = neg_proposal_list[i]
            neg_weight = neg_weight_list[i]
            gt_box = gt_points[i]
            img_meta = img_metas[i]
            filename = img_meta['filename']
            igs = cv2.imread(filename)
            import copy
            igs1 = copy.deepcopy(igs)
            boxes = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
            gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)

            for i in range(len(gt_box)):
                igs1 = cv2.rectangle(igs1, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                     color=(0, 255, 0))
                igs = cv2.rectangle(igs, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                    color=(0, 255, 0))
            for i in range(len(boxes)):
                # if neg_weight[i]:
                igs1 = cv2.rectangle(igs1, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]),
                                     color=(255, 0, 0))
            cv2.namedWindow("ims1", 0)
            cv2.resizeWindow("ims1", 640, 480)
            cv2.imshow('ims1', igs1)
            cv2.namedWindow("ims", 0)
            cv2.resizeWindow("ims", 640, 480)
            cv2.imshow('ims', igs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    # def simple_test_bboxes(self,
    #                        x,
    #                        img_metas,
    #                        proposals,
    #                        gt_bboxes,
    #                        gt_labels,
    #                        gt_anns_id,
    #                        stage,
    #                        rcnn_test_cfg,
    #                        rescale=False):
    #
    #     # get origin input shape to support onnx dynamic input shape
    #     img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    #     scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
    #
    #     rois = bbox2roi(proposals)
    #     bbox_results = self._bbox_forward(x, rois, gt_bboxes, stage)
    #     pseudo_boxes, mean_ious, box2, dynamic_weight = self.merge_box(bbox_results, proposals, torch.cat(gt_labels),
    #                                                                    gt_bboxes,
    #                                                                    img_metas, stage)
    #
    #     det_bboxes, det_labels = self.pseudobox_to_result(pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id,
    #                                                       scale_factors, rescale)
    #
    #     return det_bboxes, det_labels

    def pseudobox_to_result(self, pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id, scale_factors, rescale):
        det_bboxes = []
        det_labels = []
        batch_gt = [len(b) for b in gt_labels]
        dynamic_weight = torch.split(dynamic_weight, batch_gt)
        for i in range(len(pseudo_boxes)):
            boxes = pseudo_boxes[i]
            labels = gt_labels[i]

            if rescale and boxes.shape[0] > 0:
                scale_factor = boxes.new_tensor(scale_factors[i]).unsqueeze(0).repeat(
                    1,
                    boxes.size(-1) // 4)
                boxes /= scale_factor

            boxes = torch.cat([boxes, dynamic_weight[i].sum(dim=1, keepdim=True)], dim=1)
            gt_anns_id_single = gt_anns_id[i]
            boxes = torch.cat([boxes, gt_anns_id_single.unsqueeze(1)], dim=1)
            det_bboxes.append(boxes)
            det_labels.append(labels)
        return det_bboxes, det_labels

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
