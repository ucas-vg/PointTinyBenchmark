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
class ENoiseBox(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(ENoiseBox, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
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
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()

        ### spatial distill
        generate_proposals, _ = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                       img_meta=img_metas,
                                                       stage=0)
        adaptive_proposals = self.roi_head.forward_train_distill(x, img_metas, generate_proposals,
                                                                 gt_bboxes,
                                                                 gt_true_bboxes,
                                                                 gt_labels, )

        for stage in range(self.num_stages):
            if stage == 0:  ##CBP_stage
                # generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
                #                                                                   img_meta=img_metas)
                cascade_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                # neg_proposal_list, neg_weight_list = None, None
                # pseudo_boxes = generate_proposals
                others = None
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=stage)
                neg_proposal_list, neg_weight_list = None, None
            elif stage == 1:
                if 'reg_proposal_boxes' in bbox_results:
                    generate_proposals = bbox_results['reg_proposal_boxes']
                    proposals_valid_list = [i.new_full(i.shape[:1], 1) for i in generate_proposals]
                else:
                    generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                      img_meta=img_metas,
                                                                                      stage=stage)
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_bboxes, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            roi_losses, bbox_results = self.roi_head.forward_train(stage, x, img_metas,
                                                                   gt_bboxes if stage == 0 else pseudo_boxes,
                                                                   generate_proposals,
                                                                   proposals_valid_list,
                                                                   neg_proposal_list, neg_weight_list,
                                                                   gt_bboxes, gt_true_bboxes, gt_labels,
                                                                   cascade_weight,
                                                                   gt_bboxes_ignore, gt_masks, others=others,
                                                                   **kwargs)
            pseudo_boxes, cascade_weight, others = bbox_results['pseudo_boxes'], bbox_results['dynamic_weight'], \
                                                   bbox_results['others']

            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        batch_gt = [len(b) for b in gt_bboxes]
        dynamic_weight = torch.split(cascade_weight, batch_gt)
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
                others = None
                # cascade_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas, stage=stage)
            elif stage == 1:
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas, stage=stage)

            test_result, pseudo_boxes = self.roi_head.simple_test_pseudo(stage,
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
