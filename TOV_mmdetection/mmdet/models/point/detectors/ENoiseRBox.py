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
class ENoiseRBox(TwoStageDetector):
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
        super(ENoiseRBox, self).__init__(
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
        self.bb = 0
        self.cc = 0
        self.ee = 0
        self.ff = 0

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

        #######

        #####luxuran

        fine_proposal_cfg = self.train_cfg.get('fine_proposal', self.test_cfg.rpn)
        losses = dict()
        for stage in range(self.num_stages):
            if stage == 0:
                ### spatial distill
                with_distill = self.roi_head.bbox_head.with_distill

                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=0)

                cascade_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))

                if with_distill:
                    label_weights = cascade_weight

                    loss_distill, generate_proposals, proposals_valid_list = self.roi_head.forward_train_distill(x,
                                                                                                                 img_metas,
                                                                                                                 generate_proposals,
                                                                                                                 proposals_valid_list,
                                                                                                                 gt_bboxes,
                                                                                                                 gt_true_bboxes,
                                                                                                                 gt_labels,
                                                                                                                 gt_masks,
                                                                                                                 label_weights,
                                                                                                                 stage_mode='ori')

                    # for nn in range(3):
                    #     gt_bboxes_ = [bb.reshape(-1, 3, 4)[:, nn] for bb in gt_bboxes]
                    #     gt_labels_ = [bb.reshape(-1, 3)[:,nn] for bb in gt_labels]
                    #     cascade_weight_ = cascade_weight.reshape(-1, 3)[:, nn]
                    #     loss_distill, generate_proposals, proposals_valid_list = self.roi_head.forward_train_distill(x,
                    #                                                                                                  img_metas,
                    #                                                                                                  generate_proposals,
                    #                                                                                                  proposals_valid_list,
                    #                                                                                                  gt_bboxes_,
                    #                                                                                                  gt_true_bboxes,
                    #                                                                                                  gt_labels_,
                    #                                                                                                  cascade_weight_,
                    #                                                                                                  stage_mode='ori')
                    for key, value in loss_distill.items():
                        losses[f'ori_{key}'] = value

                    ##########luxuran
                    # generate_proposals=[bb.reshape(-1,3,4)[:,0] for bb in generate_proposals]
                    # proposals_valid_list=[bb.reshape(-1,3,1)[:,0] for bb in proposals_valid_list]
                    gt_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_bboxes]
                    gt_true_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_true_bboxes]
                    gt_labels = [bb.reshape(-1, 3)[:, 0] for bb in gt_labels]
                    cascade_weight = cascade_weight.reshape(-1, 3)[:, 0]

                    for key, value in loss_distill.items():
                        losses[f'ori_{key}'] = value
                    from mmdet.core.bbox.iou_calculators import bbox_overlaps
                    self.ee += bbox_overlaps(generate_proposals[0], gt_true_bboxes[0]).max(0)[0].sum()
                    self.ff += len(gt_bboxes[0])
                    # print('gt',self.ee / self.ff)
                else:
                    from mmdet.core.bbox.iou_calculators import bbox_overlaps
                    losses['gt_box_max'] = bbox_overlaps(gt_bboxes[0], gt_true_bboxes[0], is_aligned=True).mean()
                    # losses['ori_box_max'] = bbox_overlaps(generate_proposals[0], gt_true_bboxes[0]).max(0)[0].mean()
                    if gt_bboxes[0].shape[0]%3!=0:
                        print(gt_bboxes)
                    gt_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_bboxes]
                    gt_true_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_true_bboxes]
                    gt_labels = [bb.reshape(-1, 3)[:, 0] for bb in gt_labels]
                    cascade_weight = cascade_weight.reshape(-1, 3)[:, 0]
                    if gt_labels[0].shape[0] ==0:
                        print('a')
                    # self.ee +=  bbox_overlaps(generate_proposals[0], gt_true_bboxes[0]).max(0)[0].sum()
                    # self.ff += len(gt_bboxes[0])
                    # print('gt',self.ee / self.ff)

                from torchvision import transforms
                neg_proposal_list, neg_weight_list = None, None
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_bboxes, fine_proposal_cfg,
                                       generate_proposals,
                                       img_meta=img_metas)
            # neg_proposal_list, neg_weight_list =  gen_negative_proposals(pseudo_boxes if stage>0 else gt_bboxes, fine_proposal_cfg,
            #          generate_proposals,
            #       img_meta=img_metas)
            elif stage == 1:
                if 'reg_proposal_boxes' in bbox_results:
                    generate_proposals = bbox_results['reg_proposal_boxes']

                    proposals_valid_list = [i.new_full(i.shape[:1], 1) for i in generate_proposals]
                    # else:
                    #     generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes,
                    #                                                                       fine_proposal_cfg,
                    #                                                                       img_meta=img_metas,
                    #                                                                       stage=stage)

                    if with_distill:
                        label_weights = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                        loss_distill, generate_proposals, proposals_valid_list = self.roi_head.forward_train_distill(x,
                                                                                                                     img_metas,
                                                                                                                     generate_proposals,
                                                                                                                     proposals_valid_list,
                                                                                                                     pseudo_boxes,
                                                                                                                     gt_true_bboxes,
                                                                                                                     gt_labels,
                                                                                                                     label_weights,
                                                                                                                     stage_mode='re')
                        for key, value in loss_distill.items():
                            losses[f're_{key}'] = value
                    else:
                        losses['re_box_max'] = bbox_overlaps(generate_proposals[0], gt_true_bboxes[0]).max(0)[0].mean()

                neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_boxes, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            roi_losses, bbox_results = self.roi_head.forward_train(stage, x, img_metas,
                                                                   gt_bboxes if stage == 0 else pseudo_boxes,
                                                                   generate_proposals,
                                                                   proposals_valid_list,
                                                                   neg_proposal_list, neg_weight_list,
                                                                   gt_bboxes, gt_true_bboxes, gt_labels,
                                                                   cascade_weight,
                                                                   gt_bboxes_ignore, gt_masks, others=None,
                                                                   **kwargs)
            pseudo_boxes, cascade_weight, others = bbox_results['pseudo_boxes'], bbox_results['dynamic_weight'], \
                bbox_results['others']
            # self.bb += bbox_overlaps(pseudo_boxes[0], gt_true_bboxes[0], is_aligned=True).sum()
            # self.cc += len(gt_bboxes[0])
            # print('pred',self.bb / self.cc)

            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                       self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         pseudo_boxes,
        #         gt_labels=None,
        #         ann_weight=None,  # dynamic_weight,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg)
        #     losses.update(rpn_losses)
        #
        #     det_losses = self.roi_head.forward_train_RCNN(x, img_metas, proposal_list,
        #                                                   pseudo_boxes, gt_labels, gt_bboxes_ignore, None, gt_masks,
        #                                                   **kwargs)
        #     for key, value in det_losses.items():
        #         losses[f'det_{key}'] = value
        return losses

    def show_imgs(self, img_metas, gt_labels, pseudo_boxes,
                  gt_bboxes):
        import cv2
        for i in range(len(img_metas)):
            gt_box = gt_bboxes[i]
            pos_box = pseudo_boxes[i]

            img_meta = img_metas[i]

            pos_box = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
            gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
            ims = cv2.imread(img_meta['filename'])
            im_h, im_w, _ = img_meta['img_shape']
            ims = cv2.resize(ims, (im_w, im_h))
            for k in range(len(gt_labels[i])):
                ims = cv2.rectangle(ims, (gt_box[k, 0], gt_box[k, 1]), (gt_box[k, 2], gt_box[k, 3]),
                                    color=(0, 255, 0))
            # for i in range(len(gt_labels[i])):
            #     ims = cv2.rectangle(ims, (pos_box[i, 0], pos_box[i, 1]), (pos_box[i, 2], pos_box[i, 3]),
            #                          color=(0, 255, 0))

            for j in range(len(gt_labels[i])):
                pad_h, pad_w, _ = img_meta['pad_shape']
                im_h, im_w, _ = img_meta['img_shape']

                img = cv2.imread(img_meta['filename'])
                img = cv2.resize(img, (im_w, im_h))
                img = cv2.rectangle(img, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                    color=(0, 255, 0))
                img = cv2.rectangle(img, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                    color=(0, 255, 255))

                cv2.namedWindow("ims1", 0)
                cv2.resizeWindow("ims1", 640, 480)
                cv2.imshow('ims1', ims)
                cv2.namedWindow("im", 0)
                cv2.resizeWindow("im", im_w * 4, im_h * 4)
                cv2.imshow('im', img)
                cv2.waitKey()
                cv2.destroyAllWindows()

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
            with_distill = self.roi_head.bbox_head.with_distill
            # gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            # if stage == 0:
            #     generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
            #                                                                       img_meta=img_metas)
            # else:
            if stage == 0:

                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(gt_bboxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=0)
                gt_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_bboxes]
                gt_true_bboxes = [bb.reshape(-1, 3, 4)[:, 0] for bb in gt_true_bboxes]
                gt_labels = [bb.reshape(-1, 3)[:, 0] for bb in gt_labels]
                gt_anns_id = [bb.reshape(-1, 3)[:, 0] for bb in gt_anns_id]
                # cascade_weight = cascade_weight.reshape(-1, 3)[:, 0]
                if with_distill:
                    label_weights = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                    generate_proposals, proposals_valid_list = self.roi_head.forward_test_distill(x, img_metas,
                                                                                                  generate_proposals,
                                                                                                  proposals_valid_list,
                                                                                                  gt_bboxes,
                                                                                                  gt_true_bboxes,
                                                                                                  gt_labels,
                                                                                                  label_weights,
                                                                                                  stage_mode='ori')

            elif stage == 1:
                if 'reg_proposal_boxes' in others:
                    generate_proposals = others['reg_proposal_boxes']
                    proposals_valid_list = [i.new_full(i.shape[:1], 1) for i in generate_proposals]
                else:
                    generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes,
                                                                                      fine_proposal_cfg,
                                                                                      img_meta=img_metas,
                                                                                      stage=stage)
                    if with_distill:
                        label_weights = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                        loss_distill, generate_proposals, proposals_valid_list = self.roi_head.forward_train_distill(
                            x,
                            img_metas,
                            generate_proposals,
                            proposals_valid_list,
                            pseudo_boxes,
                            gt_true_bboxes,
                            gt_labels,
                            label_weights, stage_mode='re')

            test_result, pseudo_boxes, others = self.roi_head.simple_test_pseudo(stage,
                                                                                 x, generate_proposals,
                                                                                 proposals_valid_list,
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
