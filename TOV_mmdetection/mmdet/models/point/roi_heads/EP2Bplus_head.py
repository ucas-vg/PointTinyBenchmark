import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
# from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
import math
import copy
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from mmdet.core.point.p2b_utils.merge_bbox import merge_box
from mmdet.core.bbox.iou_calculators import bbox_overlaps


@HEADS.register_module()
class EP2BplusHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, bbox_roi_extractor, num_stages, bbox_head, bbox_head1=None, stage_modes=['CBP', 'PBR'], top_k=7,
                 with_drag=None, with_objectness=None,
                 cluster_mode='classification', with_atten=None, **kwargs):
        super(EP2BplusHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk1 = top_k  # 7
        self.topk2 = top_k  # 7
        self.stage_modes = stage_modes

        self.featmap_stride = bbox_roi_extractor.featmap_strides
        self.with_atten = with_atten
        if bbox_head1 is not None:
            self.bbox_head1 = build_head(bbox_head1)
        # self.Test_P2B_iou = Test_P2B_iou()
        # self.Test_P2B_iou_3 = Test_P2B_iou_3()
        # self.Test_P2B_iou_4 = Test_P2B_iou_4()
        # self.Test_P2B_iou_5 = Test_P2B_iou_5()
        self.cluster_mode = cluster_mode
        self.with_drag = with_drag
        self.with_objectness = with_objectness

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

    def forward_train_distill(self, x, img_metas, proposals_list, proposals_valid_list,
                              ori_bboxes, gt_true_bboxes, gt_labels, label_weights, stage_mode):
        num_gt = len(torch.cat(ori_bboxes))
        rois = bbox2roi(proposals_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        fcs_feats = self.bbox_head.forward_distill_2fcs(bbox_feats)
        if stage_mode == 'ori':
            reg_pred = self.bbox_head.forward_distill_reg_ori(fcs_feats)
        else:
            reg_pred = self.bbox_head.forward_distill_reg_re(fcs_feats)

        bboxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 4),
                                                       reg_pred.reshape(-1, 4)).reshape(num_gt, -1, 4).detach()
        bboxes_pred_ = bboxes_pred.split([len(i) for i in ori_bboxes])
        bboxes_pred_ = [i.reshape(-1, 4) for i in bboxes_pred_]

        gt = torch.cat(gt_true_bboxes).unsqueeze(1).expand(bboxes_pred.shape)
        iou = bbox_overlaps(bboxes_pred.reshape(-1, 4), gt.reshape(-1, 4), is_aligned=True).reshape(
            *bboxes_pred.shape[:2])
        iou_max, idx = iou.max(dim=1)
        mean_ious_reg_max = iou_max.mean()

        bboxes_target = self.bbox_head.bbox_coder.encode(torch.cat(proposals_list).reshape(-1, 4),
                                                         torch.cat(ori_bboxes).unsqueeze(1).expand(
                                                             bboxes_pred.shape).reshape(-1, 4)).reshape(reg_pred.shape)
        loss_distill = self.bbox_head.loss_distill(reg_pred, bboxes_target, label_weights=label_weights)
        # if self.with_drag:
        #     loss_drag = self.bbox_head.loss_drag(bboxes_pred,torch.cat(ori_bboxes),label_weights,gt_labels)
        #     for key, value in loss_drag.items():
        #         loss_distill[key] = value
        loss_distill['mean_ious_reg_max'] = mean_ious_reg_max
        # loss_distill['mean_ious_reg_max'] = iou.mean()

        return loss_distill, bboxes_pred_, proposals_valid_list

    def forward_test_distill(self, x, img_metas, proposals_list, proposals_valid_list,
                             ori_bboxes, gt_true_bboxes, gt_labels, label_weights, stage_mode):
        num_gt = len(torch.cat(ori_bboxes))
        rois = bbox2roi(proposals_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        fcs_feats = self.bbox_head.forward_distill_2fcs(bbox_feats)
        if stage_mode == 'ori':
            reg_pred = self.bbox_head.forward_distill_reg_ori(fcs_feats)
        else:
            reg_pred = self.bbox_head.forward_distill_reg_re(fcs_feats)

        bboxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 4),
                                                       reg_pred.reshape(-1, 4)).reshape(num_gt, -1, 4).detach()
        bboxes_pred_ = bboxes_pred.split([len(i) for i in ori_bboxes])
        bboxes_pred_ = [i.reshape(-1, 4) for i in bboxes_pred_]

        # self.Test_P2B_iou_5(proposals_list, gt_true_bboxes, gt_true_bboxes)
        return bboxes_pred_, proposals_valid_list

    def forward_train(self,
                      stage,
                      x,
                      img_metas,
                      proposal_list_base,
                      proposals_list,
                      proposals_valid_list,
                      neg_proposal_list,
                      neg_weight_list,
                      gt_points,
                      gt_true_bboxes,
                      gt_labels,
                      dynamic_weight,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      others=None,
                      ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # bbox head forward and loss

        bbox_results = self._bbox_forward_train(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                neg_proposal_list,
                                                neg_weight_list,
                                                gt_points, gt_true_bboxes, gt_labels, dynamic_weight,
                                                img_metas, stage, others=others)

        losses.update(bbox_results['loss_instance_mil'])

        return losses, bbox_results

    def _bbox_forward_train(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                            neg_weight_list, gt_points, gt_true_bboxes,
                            gt_labels,
                            cascade_weight,
                            img_metas, stage, others=None):
        """Run forward function and calculate loss for box head in training."""

        # self.show_box(proposals_list,neg_proposal_list,neg_weight_list, gt_points,img_metas)
        stage_mode = self.stage_modes[stage]
        rois = bbox2roi(proposals_list)
        bbox_results = self._bbox_forward(x, rois, proposal_list_base, stage, others=others)
        if self.with_objectness and stage == 1:
            object_score = bbox_results['obj_score']
            iou_score = bbox_results['iou_score']
        num_instance = bbox_results['num_instance']
        gt_labels = torch.cat(gt_labels)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)

        if neg_proposal_list is not None:
            neg_rois = bbox2roi(neg_proposal_list)
            neg_bbox_results = self._bbox_forward(x, neg_rois, None, stage)  ######stage
            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
            # if self.with_objectness and stage == 1:
            #     neg_object_score = neg_bbox_results['obj_score']

        else:
            neg_cls_scores = None
            neg_weights = None

        proposals_list_to_merge = proposals_list

        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = merge_box(bbox_results,
                                                                                             proposals_list_to_merge,
                                                                                             proposals_valid_list,
                                                                                             gt_labels,
                                                                                             gt_true_bboxes,
                                                                                             img_metas, stage_mode,
                                                                                             topk=self.topk1)
        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=dynamic_weight.sum(dim=-1))

        pseudo_boxes = torch.cat(pseudo_boxes)

        reg_box = bbox_results['bbox_pred']
        if reg_box is not None:
            bboxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 4),
                                                           reg_box.reshape(-1, 4)).reshape(reg_box.shape)
            bboxes_pred_ = bboxes_pred.split([len(i) for i in gt_points])
            bboxes_pred_ = [i.reshape(-1, 4) for i in bboxes_pred_]
            ## cal IoU with gt
            gt = torch.cat(gt_true_bboxes).unsqueeze(1).expand(reg_box.shape)
            iou = bbox_overlaps(bboxes_pred.reshape(-1, 4), gt.reshape(-1, 4), is_aligned=True).reshape(
                *reg_box.shape[:2])
            iou_max, idx = iou.max(dim=1)
            mean_ious_reg_max = [iou_max.mean()]

            bboxes_target = self.bbox_head.bbox_coder.encode(torch.cat(proposals_list).reshape(-1, 4),
                                                             pseudo_boxes.unsqueeze(1).expand(
                                                                 reg_box.shape).reshape(-1, 4)).reshape(reg_box.shape)
        else:
            bboxes_pred = None
            bboxes_target = None

        loss_instance_mil = self.bbox_head.loss_mil(stage_mode, bbox_results['cls_score'], bbox_results['ins_score'],
                                                    proposals_valid_list,
                                                    neg_cls_scores, neg_weights,
                                                    gt_labels, reg_box,
                                                    bboxes_target, label_weights=cascade_weight,
                                                    retrain_weights=dynamic_weight.sum(dim=-1))

        if self.with_objectness and stage == 1:
            a = torch.cat(proposals_list).reshape(-1, num_instance, 4)
            b = torch.cat(proposal_list_base)[:, None, :].expand(a.shape)
            pos_target = bbox_overlaps(a, b, is_aligned=True).reshape(-1, 1).detach()
            # neg_target = neg_object_score.new_full(neg_object_score.shape, 1)
            loss_objectness = self.bbox_head.loss_objectness(object_score, None, iou_score, pos_target,
                                                             neg_target=None, cascade_weight=cascade_weight)

            for key, value in loss_objectness.items():
                loss_instance_mil[key] = value
        loss_instance_mil.update({"mean_ious": mean_ious[-1]})
        loss_instance_mil.update({"s": mean_ious[0]})
        loss_instance_mil.update({"m": mean_ious[1]})
        loss_instance_mil.update({"l": mean_ious[2]})
        loss_instance_mil.update({"h": mean_ious[3]})
        if reg_box is not None:
            bbox_results['reg_proposal_boxes'] = bboxes_pred_
            # loss_instance_mil.update({"mean_ious_reg": mean_ious_reg[-1]})
            loss_instance_mil.update({"mean_ious_reg_max": mean_ious_reg_max[-1]})
            if self.with_drag:
                loss_drag = self.bbox_head.loss_drag(bboxes_pred, pseudo_boxes, cascade_weight, gt_labels)
                for key, value in loss_drag.items():
                    loss_instance_mil[key] = value

        bbox_results.update(loss_instance_mil=loss_instance_mil)

        return bbox_results

    def filter_box(self, bbox_results, proposals_list, gt_labels, gt_point, img_metas):
        num_gt = bbox_results['cls_score'].shape[0]
        num_cls = bbox_results['cls_score'].shape[-1]
        cls_score = bbox_results['cls_score'].reshape(num_gt, -1, 5, num_cls)
        cls_score = cls_score[torch.arange(num_gt), :, :, gt_labels]

        k = 3
        _, idx = cls_score.topk(k=k, dim=2)
        pps = torch.cat(proposals_list).reshape(-1, 5, 4)
        num_gt_num_gen = pps.shape[0]
        pps = pps[torch.arange(num_gt_num_gen).unsqueeze(-1), idx.reshape(-1, k)].reshape(num_gt, -1, 4)
        img_len = [i.shape[0] for i in gt_point]
        pps = torch.split(pps, img_len)
        bbox_results['cls_score'] = bbox_results['cls_score'].reshape(num_gt_num_gen, 5, num_cls)[
            torch.arange(num_gt_num_gen).unsqueeze(-1), idx.reshape(-1, k)].reshape(num_gt, -1, num_cls)
        bbox_results['ins_score'] = bbox_results['ins_score'].reshape(num_gt_num_gen, 5, num_cls)[
            torch.arange(num_gt_num_gen).unsqueeze(-1), idx.reshape(-1, k)].reshape(num_gt, -1, num_cls)
        bbox_results['bbox_pred'] = bbox_results['bbox_pred'].reshape(num_gt_num_gen, 5, 4)[
            torch.arange(num_gt_num_gen).unsqueeze(-1), idx.reshape(-1, k)].reshape(num_gt, -1, 4)
        bbox_results['bbox_feats'] = bbox_results['bbox_feats'].reshape(num_gt_num_gen, 5, -1)[
            torch.arange(num_gt_num_gen).unsqueeze(-1), idx.reshape(-1, k)].reshape(num_gt, -1,
                                                                                    bbox_results['bbox_feats'].shape[
                                                                                        -1])
        return list(pps), bbox_results

    def _bbox_forward(self, x, rois, gt_points, stage, others=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        x_fcs = self.bbox_head.forward_2fcs(bbox_feats, stage, others=others)
        cls_score, ins_score, reg_box, others = self.bbox_head.forward_head(x_fcs, stage, others=others)
        if gt_points is not None:
            gt_points = torch.cat(gt_points)
        if self.with_objectness and gt_points is not None:
            """
             x_fcs_obj: (num_gt*samples_per_gt, d)
            """
            x_fcs_obj = self.bbox_head.forward_distill_2fcs_objness(bbox_feats, stage)
            # x_fcs_obj = self.bbox_head.forward_2fcs(bbox_feats, stage)   ##### changed

            num_gt = gt_points.shape[0]
            gt_ious = bbox_overlaps(gt_points, gt_points)
            num_chose_gt = min(3, num_gt)
            _, neg_gt_idxs = gt_ious.topk(dim=1, k=num_chose_gt)  # (num_gt, num_chose_gt)
            idx = torch.arange(num_gt).to(x_fcs_obj.device)
            chosen_idxs = torch.cat((idx[:, None], neg_gt_idxs[:, 1:]), dim=1)  # (num_gt, num_chose_gt)
            assert num_gt != 0, f'num_gt = 0 {gt_points}'

            x_fcs_obj = x_fcs_obj.reshape(num_gt, -1, x_fcs_obj.shape[-1])  # (num_gt, samples_per_gt, d)
            gt_feats = x_fcs_obj.mean(1)[chosen_idxs][:, None, :]  # (num_gt, d) => (num_gt, 1, num_chose_gt, d)
            x_fcs_obj_ = x_fcs_obj[:, :, None] + gt_feats  # (num_gt, samples_per_gt, num_chose_gt, d)
            # # obj_score = self.bbox_head.forward_objectness(x_fcs_obj_, stage, others=others)
            x_fcs_iou = x_fcs_obj_.reshape(num_gt, -1, num_chose_gt, x_fcs_obj_.shape[-1])[:, :, 0, :]
            obj_score = False
            # x_fcs_iou = torch.cat([x_fcs_obj, gt_feats[:,:,0].expand(x_fcs_obj.shape)], dim=-1)
            iou_score = self.bbox_head.forward_iou(x_fcs_iou, stage, others=others)
        elif self.with_objectness and gt_points is None:
            x_fcs_obj = self.bbox_head.forward_distill_2fcs_objness(bbox_feats, stage)
            x_fcs_obj_ = x_fcs_obj
            # obj_score = self.bbox_head.forward_objectness(x_fcs_obj_ * 2, stage, others=others)
            obj_score = False
        # positive sample
        if gt_points is not None:
            #
            num_gt = gt_points.shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_points}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])
            # mil_score = F.softmax(cls_score, dim=-1) * F.softmax(posi_score, dim=-2)
            # instance_mil_score = mil_score.sum(dim=1, keepdim=False)
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats,
                num_instance=cls_score.shape[1],
                obj_score=obj_score if self.with_objectness else None,
                iou_score=iou_score if self.with_objectness and gt_points is not None else None,
                others=others)
            return bbox_results
        # negative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats,
                obj_score=obj_score if self.with_objectness else None,
                num_instance=None)
            return bbox_results

    def forward_train_RCNN(self, x, img_metas, proposals_list, gt_bboxes, gt_labels, gt_bboxes_ignore, dynamic_weight,
                           gt_masks):
        losses = dict()
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposals_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposals_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        bbox_results = self._bboxdet_forward_train(x, sampling_results,
                                                   gt_bboxes, gt_labels, dynamic_weight,  # add by fei
                                                   img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def _bboxdet_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, ann_weight,
                               img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward1(x, rois, )

        bbox_targets = self.bbox_head1.get_targets(sampling_results, gt_bboxes,
                                                   gt_labels, ann_weight, self.train_cfg)  ## add by fei
        loss_bbox = self.bbox_head1.loss(bbox_results['cls_score'],
                                         bbox_results['bbox_pred'], rois,
                                         *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward1(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head1(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bac_assigner(self, bbox, gt_bbox):
        num_gt = gt_bbox.shape[0]
        bbox = bbox.reshape(num_gt, -1, 4)
        gt_bbox = gt_bbox.unsqueeze(1).expand(bbox.shape)
        iou = bbox_overlaps(bbox, gt_bbox, is_aligned=True)
        box_weight = (iou > 0.5).long()
        _, idx = iou.max(dim=1)
        box_weight[idx] = 1
        return box_weight

    def show_box(self, proposals_list, filtered_scores, neg_proposal_list, neg_weight_list, bbox_results, gt_points,
                 img_metas):
        import cv2
        import numpy as np

        for img in range(len(img_metas)):
            pos_box = proposals_list[img]

            # neg_box = neg_proposal_list[i]
            # neg_weight = neg_weight_list[i]
            gt_box = gt_points[img]
            img_meta = img_metas[img]
            filename = img_meta['filename']
            igs = cv2.imread(filename)
            h, w, _ = img_metas[img]['img_shape']
            igs = cv2.resize(igs, (w, h))
            import copy
            igs1 = copy.deepcopy(igs)
            boxes = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
            gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
            if filtered_scores:
                filtered_score = filtered_scores[img]
                cls_score = filtered_score['cls_score']
                ins_score = filtered_score['ins_score']
                dynamic_weight = filtered_score['dynamic_weight']

            for i in range(len(gt_box)):
                igs1 = cv2.rectangle(igs1, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                     color=(0, 255, 0))
                igs = cv2.rectangle(igs, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                    color=(0, 255, 0))
            for i in range(len(boxes)):
                color = (np.random.randint(0, 255), np.random.randint(0, 255),
                         np.random.randint(0, 255))
                igs1 = copy.deepcopy(igs)

                for j in range(len(boxes[i])):
                    # if neg_weight[i]:

                    blk = np.zeros(igs1.shape, np.uint8)
                    blk = cv2.rectangle(blk, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                        color=color, thickness=-1)
                    # 得到与原图形大小形同的形状

                    igs1 = cv2.addWeighted(igs1, 1.0, blk, 0.3, 1, dst=None, dtype=None)
                    igs1 = cv2.rectangle(igs1, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                         color=color, thickness=2)
                    if filtered_scores:
                        igs1 = cv2.putText(igs1, str(cls_score[i, j]), (boxes[i, j, 0], boxes[i, j, 1]),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                        cls = cls_score[i]
                        ins = ins_score[i]
                        dyna = dynamic_weight[i]

                    # cv2.imwrite('exp/debug/'+filename,igs1)
                    cv2.namedWindow("ims1", 0)
                    cv2.resizeWindow("ims1", 2000, 1200)
                    cv2.imshow('ims1', igs1)
                    # cv2.namedWindow("ims", 0)
                    # cv2.resizeWindow("ims", 1333, 800)
                    # cv2.imshow('ims', igs)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    elif cv2.waitKey(0) & 0xFF == ord('b'):
                        break

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

    # def simple_test(self,
    #                 x,
    #                 proposal_list,
    #                 img_metas,
    #                 proposals=None,
    #                 rescale=False):
    #     """Test without augmentation."""
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #
    #     det_bboxes, det_labels = self.simple_test_bboxes(
    #         x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
    #
    #     bbox_results = [
    #         bbox2result(det_bboxes[i], det_labels[i],
    #                     self.bbox_head.num_classes)
    #         for i in range(len(det_bboxes))
    #     ]
    #
    #     if not self.with_mask:
    #         return bbox_results
    #     else:
    #         segm_results = self.simple_test_mask(
    #             x, img_metas, det_bboxes, det_labels, rescale=rescale)
    #         return list(zip(bbox_results, segm_results))

    def simple_test_pseudo(self,
                           stage,
                           x,
                           proposal_list,
                           proposals_valid_list,
                           gt_points,
                           gt_labels,
                           gt_anns_id,
                           img_metas,
                           proposals=None,
                           rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, pseudo_bboxes, others = self.simple_test_bboxes_pseudo(
            x, img_metas, proposal_list, proposals_valid_list, gt_points, gt_labels, gt_anns_id, stage, self.test_cfg,
            rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # pseudo_bboxes = [i[:, :4] for i in det_bboxes]
        return bbox_results, pseudo_bboxes, others

    def simple_test_bboxes_pseudo(self,
                                  x,
                                  img_metas,
                                  proposals,
                                  proposals_valid_list,
                                  gt_bboxes,
                                  gt_labels,
                                  gt_anns_id,
                                  stage,
                                  rcnn_test_cfg,
                                  rescale=False):
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, gt_bboxes, stage)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)
        stage_mode = self.stage_modes[stage]
        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = merge_box(bbox_results,
                                                                                             proposals,
                                                                                             proposals_valid_list,
                                                                                             torch.cat(gt_labels),
                                                                                             gt_bboxes,
                                                                                             img_metas, stage_mode,
                                                                                             topk=self.topk1)
        # if stage == 0:
        #     self.Test_P2B_iou_3(proposals, gt_bboxes, pseudo_boxes)
        # elif stage == 1:
        #     self.Test_P2B_iou_4(proposals, gt_bboxes, pseudo_boxes)

        reg_box = bbox_results['bbox_pred']
        if reg_box is not None:
            bboxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals).reshape(-1, 4),
                                                           reg_box.reshape(-1, 4)).reshape(reg_box.shape)
            if bbox_results['others'] is None:
                bbox_results['others'] = {}
                bboxes_pred_ = bboxes_pred.split([len(i) for i in gt_bboxes])
                bboxes_pred_ = [i.reshape(-1, 4) for i in bboxes_pred_]

                bbox_results['others']['reg_proposal_boxes'] = bboxes_pred_

            # if self.cluster_mode == 'upper':
            #     from mmdet.core.bbox.iou_calculators import bbox_overlaps
            #     gt = torch.cat(gt_bboxes).unsqueeze(1).expand(reg_box.shape)
            #     iou = bbox_overlaps(bboxes_pred.reshape(-1, 4), gt.reshape(-1, 4), is_aligned=True)
            #     iou = iou.reshape(*reg_box.shape[:2])
            #     iou_max, idx = iou.max(dim=1)
            #     pseudo_boxes_reg = bboxes_pred[torch.arange(len(bboxes_pred)), idx]
            #     bbox_results['pseudo_boxes'] = pseudo_boxes_reg.split([len(i) for i in gt_bboxes])
            #
            #     # self.Test_P2B_iou_2(bboxes_pred.split([len(i) for i in gt_bboxes]), gt_bboxes, pseudo_boxes)
            #     pseudo_boxes = bbox_results['pseudo_boxes']

            bboxes_pred = None
            bboxes_target = None

        pseudo_boxes_out = copy.deepcopy(pseudo_boxes)

        det_bboxes, det_labels = self.pseudobox_to_result(pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id,
                                                          scale_factors, rescale)

        if bbox_results['others'] is not None:
            return det_bboxes, det_labels, pseudo_boxes_out, bbox_results['others']
        else:
            return det_bboxes, det_labels, pseudo_boxes_out, None

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
            gt_anns_id_single = gt_anns_id[i].float()
            boxes = torch.cat([boxes, gt_anns_id_single.unsqueeze(1)], dim=1)
            det_bboxes.append(boxes)
            det_labels.append(labels)
        return det_bboxes, det_labels

    def test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.test_bboxes(x, img_metas,
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
            segm_results = self.test_mask(x, img_metas, det_bboxes,
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

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # The length of proposals of different batches may be different.
        # In order to form a batch, a padding operation is required.
        max_size = max([proposal.size(0) for proposal in proposals])
        # padding to form a batch
        for i, proposal in enumerate(proposals):
            supplement = proposal.new_full(
                (max_size - proposal.size(0), proposal.size(1)), 0)
            proposals[i] = torch.cat((supplement, proposal), dim=0)
        rois = torch.stack(proposals, dim=0)

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward1(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        # remove padding, ignore batch_index when calculating mask
        supplement_mask = rois.abs()[..., 1:].sum(dim=-1) == 0
        cls_score[supplement_mask, :] = 0

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.reshape(batch_size,
                                              num_proposals_per_img,
                                              bbox_pred.size(-1))
                bbox_pred[supplement_mask, :] = 0
            else:
                # TODO: Looking forward to a better way
                # TODO move these special process to a corresponding head
                # For SABL
                bbox_preds = self.bbox_head1.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                # apply bbox post-processing to each image individually
                det_bboxes = []
                det_labels = []
                for i in range(len(proposals)):
                    # remove padding
                    supplement_mask = proposals[i].abs().sum(dim=-1) == 0
                    for bbox in bbox_preds[i]:
                        bbox[supplement_mask] = 0
                    det_bbox, det_label = self.bbox_head1.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_preds[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)
                return det_bboxes, det_labels
        else:
            bbox_pred = None

        return self.bbox_head1.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shapes,
            scale_factors,
            rescale=rescale,
            cfg=rcnn_test_cfg)


class Test_P2B_iou(nn.Module):
    def __init__(self, dim_in=256):
        super(Test_P2B_iou, self).__init__()
        self.sum_iou = 0
        self.sum_num = 0
        self.iou_bin = torch.zeros([50]).int()

    def forward(self, pseudo_boxes, gt_bboxes, mode='best_proposal_iou'):
        num_gt = torch.cat(gt_bboxes).shape[0]
        pseudo_boxes = torch.cat(pseudo_boxes).reshape(num_gt, -1, 4)
        gt_bboxes = torch.cat(gt_bboxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        iou1 = bbox_overlaps(pseudo_boxes, gt_bboxes, is_aligned=True)
        max_iou, _ = iou1.max(dim=1)
        import math
        self.sum_iou += max_iou.sum()
        self.sum_num += num_gt
        for i in max_iou:
            bin = math.floor(i / 0.02)

            if bin == 50:
                bin = 49
            self.iou_bin[bin] += 1
        print(self.iou_bin)
        print(self.sum_iou / self.sum_num)


class Test_P2B_iou_2(nn.Module):
    def __init__(self, dim_in=256):
        super(Test_P2B_iou_2, self).__init__()
        self.iou_minus_norm_delet_bg = torch.zeros(40).to('cuda:0')
        self.iou_minus_delet_bg = torch.zeros(40).to('cuda:0')
        self.iou_minus_norm = torch.zeros(40).to('cuda:0')
        self.iou_minus = torch.zeros(40).to('cuda:0')
        self.iou1 = torch.zeros(20).to('cuda:0')
        self.iou2 = torch.zeros(20).to('cuda:0')
        self.iou1_max = torch.zeros(20).to('cuda:0')
        self.iou2_max = torch.zeros(20).to('cuda:0')
        self.iou_gt_mb = torch.zeros(20).to('cuda:0')

    def forward(self, pseudo_boxes, gt_bboxes, merge_boxes, mode='best_proposal_iou'):
        num_gt = torch.cat(gt_bboxes).shape[0]
        pseudo_boxes = torch.cat(pseudo_boxes).reshape(num_gt, -1, 4)
        merge_boxes = torch.cat(merge_boxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        gt_bboxes = torch.cat(gt_bboxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        iou1 = bbox_overlaps(pseudo_boxes, gt_bboxes, is_aligned=True)
        iou2 = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        iou1 = iou1.reshape(*pseudo_boxes.shape[:2])
        iou2 = iou2.reshape(*pseudo_boxes.shape[:2])
        max_iou1, _ = iou1.max(dim=1)
        max_iou2, _ = iou2.max(dim=1)
        iou_gt_mb = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        ####
        iou_minus = iou1 - iou2
        iou_minus_norm = iou_minus / iou_gt_mb

        self.iou_minus_norm += torch.histc(iou_minus_norm.reshape(-1), 40, -1, 1)
        self.iou_minus += torch.histc(iou_minus.reshape(-1), 40, -1, 1)
        self.iou1 += torch.histc(iou1.reshape(-1), 20, 0, 1)
        self.iou2 += torch.histc(iou2.reshape(-1), 20, 0, 1)
        self.iou1_max += torch.histc(max_iou1.reshape(-1), 20, 0, 1)
        self.iou2_max += torch.histc(max_iou2.reshape(-1), 20, 0, 1)
        self.iou_gt_mb += torch.histc(iou_gt_mb[:, 0].reshape(-1), 20, -1, 1)

        iou_minus[((iou1 < 0.5) * (iou2 < 0.5))] = 100
        iou_minus_norm[((iou1 < 0.5) * (iou2 < 0.5))] = 100

        self.iou_minus_norm_delet_bg += torch.histc(iou_minus_norm.reshape(-1), 40, -1, 1)
        self.iou_minus_delet_bg += torch.histc(iou_minus.reshape(-1), 40, -1, 1)
        f = open('iou.txt', 'w')
        f.writelines(
            str(self.iou_minus_norm) + '\n' + str(self.iou_minus) + '\n' + str(self.iou1) + '\n' + str(self.iou2) + '\n'
            + str(self.iou1_max) + '\n' + str(self.iou2_max) + '\n' + str(self.iou_gt_mb) + '\n' + str(
                self.iou_minus_norm_delet_bg) + '\n' + str(self.iou_minus_delet_bg))
        f.close()


class Test_P2B_iou_3(nn.Module):
    def __init__(self, dim_in=256):
        super(Test_P2B_iou_3, self).__init__()
        # self.iou_minus_norm_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_norm = torch.zeros(80).to('cuda:0')
        # self.iou_minus = torch.zeros(80).to('cuda:0')
        self.iou1 = torch.zeros(50).to('cuda:0')
        # self.iou2 = torch.zeros(40).to('cuda:0')
        self.iou1_max = torch.zeros(50).to('cuda:0')
        # self.iou2_max = torch.zeros(40).to('cuda:0')
        self.iou1_mean = torch.zeros(50).to('cuda:0')
        # self.iou2_mean = torch.zeros(40).to('cuda:0')
        self.iou_gt_mb = torch.zeros(50).to('cuda:0')

        self.top10_mean = torch.zeros(50).to('cuda:0')

    def forward(self, pseudo_boxes, gt_bboxes, merge_boxes, mode='best_proposal_iou'):
        num_gt = torch.cat(gt_bboxes).shape[0]
        pseudo_boxes = torch.cat(pseudo_boxes).reshape(num_gt, -1, 4)
        merge_boxes = torch.cat(merge_boxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        gt_bboxes = torch.cat(gt_bboxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        iou1 = bbox_overlaps(pseudo_boxes, gt_bboxes, is_aligned=True)
        iou2 = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        iou1 = iou1.reshape(*pseudo_boxes.shape[:2])
        iou2 = iou2.reshape(*pseudo_boxes.shape[:2])
        top10, _ = iou1.topk(10, dim=1)
        top10_mean = top10.mean(1)
        max_iou1, _ = iou1.max(dim=1)
        max_iou2, _ = iou2.max(dim=1)
        mean_iou1 = iou1.mean(dim=1)
        mean_iou2 = iou2.mean(dim=1)
        iou_gt_mb = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        ####
        iou_minus = iou1 - iou2
        iou_minus_norm = iou_minus / iou_gt_mb

        # self.iou_minus_norm += torch.histc(iou_minus_norm.reshape(-1), 80, -1, 1)
        # self.iou_minus += torch.histc(iou_minus.reshape(-1), 80, -1, 1)
        self.iou1 += torch.histc(iou1.reshape(-1), 50, 0, 1)
        # self.iou2 += torch.histc(iou2.reshape(-1), 40, 0, 1)
        self.iou1_max += torch.histc(max_iou1.reshape(-1), 50, 0, 1)
        # self.iou2_max += torch.histc(max_iou2.reshape(-1), 40, 0, 1)
        self.iou1_mean += torch.histc(mean_iou1.reshape(-1), 50, 0, 1)
        # self.iou2_mean += torch.histc(mean_iou2.reshape(-1), 40, 0, 1)
        self.iou_gt_mb += torch.histc(iou_gt_mb[:, 0].reshape(-1), 50, -1, 1)
        self.top10_mean += torch.histc(top10_mean.reshape(-1), 50, 0, 1)

        iou_minus[((iou1 < 0.5) * (iou2 < 0.5))] = 100
        iou_minus_norm[((iou1 < 0.5) * (iou2 < 0.5))] = 100

        # self.iou_minus_norm_delet_bg += torch.histc(iou_minus_norm.reshape(-1), 40, -1, 1)
        # self.iou_minus_delet_bg += torch.histc(iou_minus.reshape(-1), 40, -1, 1)
        f = open('iou_stage1.txt', 'w')
        f.writelines(
            str(self.iou1) + '\n' + str(self.iou1_max) + '\n' + str(self.iou1_mean) + '\n' + str(
                self.iou_gt_mb) + '\n' + str(self.top10_mean) + '\n')
        f.close()


class Test_P2B_iou_4(nn.Module):
    def __init__(self, dim_in=256):
        super(Test_P2B_iou_4, self).__init__()
        # self.iou_minus_norm_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_norm = torch.zeros(80).to('cuda:0')
        # self.iou_minus = torch.zeros(80).to('cuda:0')
        self.iou1 = torch.zeros(50).to('cuda:0')
        # self.iou2 = torch.zeros(40).to('cuda:0')
        self.iou1_max = torch.zeros(50).to('cuda:0')
        # self.iou2_max = torch.zeros(40).to('cuda:0')
        self.iou1_mean = torch.zeros(50).to('cuda:0')
        self.top10_mean = torch.zeros(50).to('cuda:0')
        # self.iou2_mean = torch.zeros(40).to('cuda:0')
        self.iou_gt_mb = torch.zeros(50).to('cuda:0')
        # self.a=0
        # self.b=0

    def forward(self, pseudo_boxes, gt_bboxes, merge_boxes, mode='best_proposal_iou'):
        num_gt = torch.cat(gt_bboxes).shape[0]
        pseudo_boxes = torch.cat(pseudo_boxes).reshape(num_gt, -1, 4)
        merge_boxes = torch.cat(merge_boxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        gt_bboxes = torch.cat(gt_bboxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        iou1 = bbox_overlaps(pseudo_boxes, gt_bboxes, is_aligned=True)
        iou2 = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        iou1 = iou1.reshape(*pseudo_boxes.shape[:2])
        iou2 = iou2.reshape(*pseudo_boxes.shape[:2])
        max_iou1, _ = iou1.max(dim=1)
        # self.a+=max_iou1.sum()
        # self.b+=len(max_iou1)
        # print(self.a/self.b)
        max_iou2, _ = iou2.max(dim=1)
        top10, _ = iou1.topk(10, dim=1)
        top10_mean = top10.mean(1)
        mean_iou1 = iou1.mean(dim=1)
        mean_iou2 = iou2.mean(dim=1)
        iou_gt_mb = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        ####
        iou_minus = iou1 - iou2
        iou_minus_norm = iou_minus / iou_gt_mb

        # self.iou_minus_norm += torch.histc(iou_minus_norm.reshape(-1), 80, -1, 1)
        # self.iou_minus += torch.histc(iou_minus.reshape(-1), 80, -1, 1)
        self.iou1 += torch.histc(iou1.reshape(-1), 50, 0, 1)
        # self.iou2 += torch.histc(iou2.reshape(-1), 40, 0, 1)
        self.iou1_max += torch.histc(max_iou1.reshape(-1), 50, 0, 1)
        # self.iou2_max += torch.histc(max_iou2.reshape(-1), 40, 0, 1)
        self.top10_mean += torch.histc(top10_mean.reshape(-1), 50, 0, 1)
        self.iou1_mean += torch.histc(mean_iou1.reshape(-1), 50, 0, 1)
        # self.iou2_mean += torch.histc(mean_iou2.reshape(-1), 40, 0, 1)
        self.iou_gt_mb += torch.histc(iou_gt_mb[:, 0].reshape(-1), 50, -1, 1)

        iou_minus[((iou1 < 0.5) * (iou2 < 0.5))] = 100
        iou_minus_norm[((iou1 < 0.5) * (iou2 < 0.5))] = 100

        # self.iou_minus_norm_delet_bg += torch.histc(iou_minus_norm.reshape(-1), 40, -1, 1)
        # self.iou_minus_delet_bg += torch.histc(iou_minus.reshape(-1), 40, -1, 1)
        f = open('iou_stage2.txt', 'w')
        f.writelines(
            str(self.iou1) + '\n' + str(self.iou1_max) + '\n' + str(self.iou1_mean) + '\n' + str(
                self.iou_gt_mb) + '\n' + str(self.top10_mean) + '\n')
        f.close()


class Test_P2B_iou_5(nn.Module):
    def __init__(self, dim_in=256):
        super(Test_P2B_iou_5, self).__init__()
        # self.iou_minus_norm_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_delet_bg = torch.zeros(80).to('cuda:0')
        # self.iou_minus_norm = torch.zeros(80).to('cuda:0')
        # self.iou_minus = torch.zeros(80).to('cuda:0')
        self.iou1 = torch.zeros(50).to('cuda:0')
        # self.iou2 = torch.zeros(40).to('cuda:0')
        self.iou1_max = torch.zeros(50).to('cuda:0')
        # self.iou2_max = torch.zeros(40).to('cuda:0')
        self.iou1_mean = torch.zeros(50).to('cuda:0')
        self.top10_mean = torch.zeros(50).to('cuda:0')
        # self.iou2_mean = torch.zeros(40).to('cuda:0')
        self.iou_gt_mb = torch.zeros(50).to('cuda:0')

    def forward(self, pseudo_boxes, gt_bboxes, merge_boxes, mode='best_proposal_iou'):
        num_gt = torch.cat(gt_bboxes).shape[0]
        pseudo_boxes = torch.cat(pseudo_boxes).reshape(num_gt, -1, 4)
        merge_boxes = torch.cat(merge_boxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        gt_bboxes = torch.cat(gt_bboxes).reshape(num_gt, 1, 4).expand(pseudo_boxes.shape)
        iou1 = bbox_overlaps(pseudo_boxes, gt_bboxes, is_aligned=True)
        iou2 = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        iou1 = iou1.reshape(*pseudo_boxes.shape[:2])
        iou2 = iou2.reshape(*pseudo_boxes.shape[:2])
        max_iou1, _ = iou1.max(dim=1)
        max_iou2, _ = iou2.max(dim=1)
        mean_iou1 = iou1.mean(dim=1)
        top10, _ = iou1.topk(10, dim=1)
        top10_mean = top10.mean(1)
        mean_iou2 = iou2.mean(dim=1)
        iou_gt_mb = bbox_overlaps(gt_bboxes, merge_boxes, is_aligned=True)
        ####
        iou_minus = iou1 - iou2
        iou_minus_norm = iou_minus / iou_gt_mb

        # self.iou_minus_norm += torch.histc(iou_minus_norm.reshape(-1), 80, -1, 1)
        # self.iou_minus += torch.histc(iou_minus.reshape(-1), 80, -1, 1)
        self.iou1 += torch.histc(iou1.reshape(-1), 50, 0, 1)
        # self.iou2 += torch.histc(iou2.reshape(-1), 40, 0, 1)
        self.iou1_max += torch.histc(max_iou1.reshape(-1), 50, 0, 1)
        # self.iou2_max += torch.histc(max_iou2.reshape(-1), 40, 0, 1)
        self.iou1_mean += torch.histc(mean_iou1.reshape(-1), 50, 0, 1)
        self.top10_mean += torch.histc(top10_mean.reshape(-1), 50, 0, 1)
        # self.iou2_mean += torch.histc(mean_iou2.reshape(-1), 40, 0, 1)
        self.iou_gt_mb += torch.histc(iou_gt_mb[:, 0].reshape(-1), 50, -1, 1)

        iou_minus[((iou1 < 0.5) * (iou2 < 0.5))] = 100
        iou_minus_norm[((iou1 < 0.5) * (iou2 < 0.5))] = 100

        # self.iou_minus_norm_delet_bg += torch.histc(iou_minus_norm.reshape(-1), 40, -1, 1)
        # self.iou_minus_delet_bg += torch.histc(iou_minus.reshape(-1), 40, -1, 1)
        f = open('iou_ori.txt', 'w')
        f.writelines(
            str(self.iou1) + '\n' + str(self.iou1_max) + '\n' + str(self.iou1_mean) + '\n' + str(
                self.iou_gt_mb) + '\n' + str(self.top10_mean) + '\n')
        f.close()


class Pts_attention(nn.Module):
    def __init__(self, dim_in=256):
        super(Pts_attention, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.fc2 = nn.Linear(dim_in, dim_in)
        self.fc3 = nn.Linear(dim_in, dim_in)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x, x_pt):
        a = x
        a = a.reshape(*a.shape[:2], -1)
        b = x_pt
        a = a.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        a = self.fc1(a)
        b = self.fc2(b)
        c = self.fc3(b)

        score = torch.matmul(b, a.permute(0, 2, 1))
        score = score / math.sqrt(b.shape[-1])
        atten_score = score.permute(0, 2, 1).softmax(-1)
        atten_map = torch.matmul(atten_score, c)
        out = x + atten_map.permute(0, 2, 1)
        return out
