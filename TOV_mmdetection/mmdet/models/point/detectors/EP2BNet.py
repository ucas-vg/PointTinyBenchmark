import copy
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import numpy as np
from mmdet.models.builder import build_head
from mmdet.core import bbox_cxcywh_to_xyxy
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.point.p2b_utils.box_sampler import CBP_proposals_from_cfg, gen_negative_proposals, PBR_proposals_from_cfg,imbalance_proposal


@DETECTORS.register_module()
class EP2BNet(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 mask_branch=None,
                 mask_head=None,
                 dense_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(EP2BNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        if dense_head is not None:
            self.with_dense_head = True
            self.dense_head = build_head(dense_head)
        else:
            self.with_dense_head = False
        if mask_branch is not None:
            self.with_mask_branch = True
            self.mask_branch = build_head(mask_branch)
        else:
            self.with_mask_branch = False
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)
        if mask_head is not None:
            self.with_mask_head = True
            self.mask_head = build_head(mask_head)
        # self.with_rpn=True

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ann_weight=None,
                      **kwargs):
        x = self.extract_feat(img)
        # print(x[0][0].shape)
        # print(x[3][0].shape)
        # print(len(x))
        # print(img_metas)
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]





        for stage in range(self.num_stages):
            if stage == 0:  ##CBP_stage
                generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = generate_proposals
                if ann_weight is not None:
                    proposals_valid_list = imbalance_proposal(proposals_valid_list, ann_weight)
            else:  ##PBR_stage
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas,
                                                                                   stage=stage)
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
                if ann_weight is not None:
                    proposals_valid_list = imbalance_proposal(proposals_valid_list, ann_weight)
            roi_losses, pseudo_boxes, dynamic_weight,others = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_points,gt_true_bboxes,gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   **kwargs)
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        batch_gt = [len(b) for b in gt_bboxes]
        dynamic_weight = torch.split(dynamic_weight, batch_gt)
        # print(dynamic_weight？.s)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                pseudo_boxes,
                gt_labels=None,
                ann_weight=None,  # dynamic_weight,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            roi_losses = self.roi_head.forward_train('with_rpn', x, img_metas, None, proposal_list, None, None, None,
                                                     None,
                                                     pseudo_boxes, gt_labels, None,  # dynamic_weight,  ## add by fei
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        for key, value in roi_losses.items():
            losses[f'det_{key}'] = value
        return losses
