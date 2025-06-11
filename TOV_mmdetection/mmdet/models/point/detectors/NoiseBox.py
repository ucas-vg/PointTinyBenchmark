import copy

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import numpy as np
from mmdet.models.builder import build_head
from mmdet.core import bbox_cxcywh_to_xyxy
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.point.p2b_utils.box_sampler import CBP_proposals_from_cfg, gen_negative_proposals, \
    PBR_proposals_from_cfg


@DETECTORS.register_module()
class NoiseBox(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(NoiseBox, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]

        for stage in range(self.num_stages):
            if stage == 0:  ##CBP_stage
                # generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
                #                                                                   img_meta=img_metas)
                cascade_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                # neg_proposal_list, neg_weight_list = None, None
                # pseudo_boxes = generate_proposals
                neg_proposal_list, neg_weight_list = None, None
                others=None
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=stage)
            elif stage == 1:
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=stage)
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)

            roi_losses, pseudo_boxes, cascade_weight,others = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   gt_bboxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_points, gt_true_bboxes, gt_labels,
                                                                                   cascade_weight,
                                                                                   gt_bboxes_ignore, gt_masks,others=others,
                                                                                   **kwargs)
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                cascade_weight_out = cascade_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value
        return losses

    def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id, gt_true_bboxes, gt_labels,
                    gt_bboxes_ignore=None, proposals=None, rescale=False):
        """Test without augmentation."""
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        for stage in range(self.num_stages):

            # gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            # if stage == 0:
            #     generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
            #                                                                       img_meta=img_metas)
            # else:
            if stage == 0:
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas, stage=stage)
            elif stage == 1:
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas, stage=stage)

            test_result, pseudo_boxes = self.roi_head.simple_test(stage,
                                                                  x, generate_proposals, proposals_valid_list,
                                                                  gt_true_bboxes, gt_labels,
                                                                  gt_anns_id,
                                                                  img_metas,
                                                                  rescale=rescale)
        return test_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # modified by hui #####################################
        if self.test_cfg.rcnn.get('do_tile_as_aug', False):
            x = self.extract_feats(imgs)
            proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
            return self.roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)
        else:
            return self.tile_aug_test(imgs, img_metas, rescale)
        ####################################################
