import copy

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head


def hbb_proposalbox_aug(proposals_box, num_gen, num_boxes, iou_threshhold, img_meta, if_return_feat=None,
                        proposal_feat=None):
    img_h, img_w, _ = img_meta['img_shape']

    ### generate proposals
    width = proposals_box[:, 2, None] - proposals_box[:, 0, None]
    height = proposals_box[:, 3, None] - proposals_box[:, 1, None]
    xl_l = proposals_box[:, 0, None] - ((1 - iou_threshhold) / iou_threshhold) * width
    yl_l = proposals_box[:, 1, None] - ((1 - iou_threshhold) / iou_threshhold) * height
    xl_r = proposals_box[:, 0, None] + (1 - iou_threshhold) * width
    yl_r = proposals_box[:, 1, None] + (1 - iou_threshhold) * height

    xr_l = proposals_box[:, 2, None] - (1 - iou_threshhold) * width
    yr_l = proposals_box[:, 3, None] - (1 - iou_threshhold) * height
    xr_r = proposals_box[:, 2, None] + (1 - iou_threshhold) / iou_threshhold * width
    yr_r = proposals_box[:, 3, None] + (1 - iou_threshhold) / iou_threshhold * height

    rand = torch.rand((proposals_box.size(0), num_gen, 4)).to(proposals_box.device)

    x1 = torch.clamp(xl_l + (xl_r - xl_l) * rand[:, :, 0], -img_w * 0.2, img_w * 1.2)
    y1 = torch.clamp(yl_l + (yl_r - yl_l) * rand[:, :, 1], -img_h * 0.2, img_h * 1.2)
    x2 = torch.clamp(xr_l + (xr_r - xr_l) * rand[:, :, 2], -img_w * 0.2, img_w * 1.2)
    y2 = torch.clamp(yr_l + (yr_r - yr_l) * rand[:, :, 3], -img_h * 0.2, img_h * 1.2)

    gen_bboxes = torch.stack([x1, y1, x2, y2], dim=0).permute(1, 2, 0)
    proposals_box_ = proposals_box.unsqueeze(1)
    # print(bbox_overlaps(gen_bboxes, )+ 'bboxes_).squeeze(2))

    overlap_of_gt_ = abs(bbox_overlaps(gen_bboxes, proposals_box_).squeeze(2) - iou_threshhold)
    # print(overlap_of_gt_)
    # print(torch.max(overlap_of_gt_,dim=1)[0])
    # print(torch.min(overlap_of_gt_,dim=1)[0])
    ###############topk

    topk_value, topk_index = torch.topk(overlap_of_gt_, num_boxes, dim=1, largest=False)
    # print(topk_value)
    # ###############random
    topk_index = torch.randint(0, num_gen, [gen_bboxes.shape[0], num_boxes])

    # print(overlap_of_gt_[torch.tensor((range(len(gen_bboxes)))).unsqueeze(1).long(), topk_index])
    generate_proposal_boxes = gen_bboxes[torch.tensor((range(len(gen_bboxes)))).unsqueeze(1).long(), topk_index]
    generate_proposal_boxes_list = generate_proposal_boxes.view([-1, 4])
    # generate_proposal_boxes_list = torch.cat([proposals_box,generate_proposal_boxes_list])

    if if_return_feat:
        assert proposal_feat is not None
        aug_ori_index = torch.tensor(range(0, len(proposals_box)))
        aug_ori_index = torch.repeat_interleave(aug_ori_index, num_boxes, dim=0)

        gen_proposal_feat = proposal_feat[:, aug_ori_index, ...]
        return generate_proposal_boxes_list, gen_proposal_feat

    return generate_proposal_boxes_list


def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    if 'cut_mode' in proposal_cfg:
        cut_mode = proposal_cfg['cut_mode']
    else:
        cut_mode = 'symmetry'

    base_proposal_list = []
    aug_proposal_list = []
    proposals_valid_list = []
    for i in range(len(gt_points)):
        img_h, img_w, _ = img_meta[i]['img_shape']
        base = min(img_w, img_h) / 100
        base_proposals = []
        for scale in base_scales:
            scale = scale * base
            for ratio in base_ratios:
                base_proposals.append(gt_points[i].new_tensor([[scale * ratio, scale / ratio]]))

        base_proposals = torch.cat(base_proposals)
        base_proposals = base_proposals.repeat((len(gt_points[i]), 1))
        base_center = torch.repeat_interleave(gt_points[i], len(base_scales) * len(base_ratios), dim=0)

        if shake_ratio is not None:
            base_x_l = base_center[:, 0] - shake_ratio * base_proposals[:, 0]
            base_x_r = base_center[:, 0] + shake_ratio * base_proposals[:, 0]
            base_y_t = base_center[:, 1] - shake_ratio * base_proposals[:, 1]
            base_y_d = base_center[:, 1] + shake_ratio * base_proposals[:, 1]
            if cut_mode is not None:
                base_x_l = torch.clamp(base_x_l, 1, img_w - 1)
                base_x_r = torch.clamp(base_x_r, 1, img_w - 1)
                base_y_t = torch.clamp(base_y_t, 1, img_h - 1)
                base_y_d = torch.clamp(base_y_d, 1, img_h - 1)

            base_center_l = torch.stack([base_x_l, base_center[:, 1]], dim=1)
            base_center_r = torch.stack([base_x_r, base_center[:, 1]], dim=1)
            base_center_t = torch.stack([base_center[:, 0], base_y_t], dim=1)
            base_center_d = torch.stack([base_center[:, 0], base_y_d], dim=1)

            shake_mode = 0
            if shake_mode == 0:
                base_proposals = base_proposals.unsqueeze(1).repeat((1, 5, 1))
            elif shake_mode == 1:
                base_proposals_l = torch.stack([((base_center[:, 0] - base_x_l) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_r = torch.stack([((base_x_r - base_center[:, 0]) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_t = torch.stack([base_proposals[:, 0],
                                                ((base_center[:, 1] - base_y_t) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals_d = torch.stack([base_proposals[:, 0],
                                                ((base_y_d - base_center[:, 1]) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals = torch.stack(
                    [base_proposals, base_proposals_l, base_proposals_r, base_proposals_t, base_proposals_d], dim=1)

            base_center = torch.stack([base_center, base_center_l, base_center_r, base_center_t, base_center_d], dim=1)

        if cut_mode == 'symmetry':
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * base_center[..., 0])
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * (img_w - base_center[..., 0]))
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * base_center[..., 1])
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * (img_h - base_center[..., 1]))

        base_proposals = torch.cat([base_center, base_proposals], dim=-1)
        base_proposals = base_proposals.reshape(-1, 4)
        base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
        proposals_valid = base_proposals.new_full(
            (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        if cut_mode == 'symmetry':
            proposals_valid_list.append(proposals_valid)
        if cut_mode == 'clamp':
            base_proposals[..., 0:4:2] = torch.clamp(base_proposals[..., 0:4:2], 0, img_w)
            base_proposals[..., 1:4:2] = torch.clamp(base_proposals[..., 1:4:2], 0, img_h)
            proposals_valid_list.append(proposals_valid)
        # proposals_valid = base_proposals.new_full(
        #     (*base_proposals.shape[0:2], 1), 1, dtype=torch.long).reshape(-1, 1)
        elif cut_mode == 'ignore':
            img_xyxy = base_proposals.new_tensor([0, 0, img_w, img_h])
            iof_in_img = bbox_overlaps(base_proposals, img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
        elif cut_mode is None:
            proposals_valid_list.append(proposals_valid)
        base_proposal_list.append(base_proposals)

        # aug_proposals = hbb_proposalbox_aug(base_proposals, num_gen, num_gen, iou_threshhold=0.7, img_meta=img_meta[i])
        # aug_proposal_list.append(aug_proposals)
    #         print(proposals_valid_list)
    return base_proposal_list, proposals_valid_list


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        ## -0.1 w -> 1.1 w
        ## -0.1 h -> 1.1 h
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_bboxes = torch.stack([x1, y1, x2, y2], dim=1).to(gt_points[0].device)
        gt_point = gt_points[i]
        gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
        iou = bbox_overlaps(neg_bboxes, pos_box)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

        ## ensure at least one:
        # _, flag = iou.sum(dim=-1).min(dim=0)
        # neg_weight[flag] = 1

        # iou2 = bbox_overlaps(neg_bboxes, gt_min_box)
        # ne2 = ((iou2 == 0).sum(dim=1) == iou2.shape[1])
        # neg_weight = neg_weight

        ## filter invalid
        # valid = torch.nonzero(neg_weight)
        # neg_bboxes = neg_bboxes[valid].squeeze(1)
        # neg_weight = neg_weight[valid].squeeze(1)

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    #         print(sum(torch.cat(neg_weight_list)>0))
    #         assert(sum(torch.cat(neg_weight_list)>0)>0)
    return neg_proposal_list, neg_weight_list


def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    if isinstance(fine_proposal_cfg['base_ratios'], tuple):
        base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
    else:
        base_ratios = fine_proposal_cfg['base_ratios']
        shake_ratio = fine_proposal_cfg['shake_ratio']
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
                    base_boxes_[:, 2] *= ratio_w
                    base_boxes_[:, 3] *= ratio_h
                    base_boxes_ = bbox_cxcywh_to_xyxy(base_boxes_)
                    pps.append(base_boxes_.unsqueeze(1))
            pps_old = torch.cat(pps, dim=1)
            if shake_ratio is not None:
                pps_new = []
                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
                for ratio in shake_ratio:
                    pps = bbox_xyxy_to_cxcywh(pps_old)
                    pps_center = pps[:, :, :2]
                    pps_wh = pps[:, :, 2:4]
                    pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                    pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                    pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
                    pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
                    pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    # pps_center_lt = torch.stack([pps_x_l, pps_y_t], dim=-1)
                    # pps_center_rt = torch.stack([pps_x_r, pps_y_t], dim=-1)
                    # pps_center_ld = torch.stack([pps_x_l, pps_y_d], dim=-1)
                    # pps_center_rd = torch.stack([pps_x_r, pps_y_d], dim=-1)
                    # pps_center = torch.stack([pps_center_lt, pps_center_rt, pps_center_ld, pps_center_rd], dim=2)
                    pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                    pps = torch.cat([pps_center, pps_wh], dim=-1)
                    pps = pps.reshape(pps.shape[0], -1, 4)
                    pps = bbox_cxcywh_to_xyxy(pps)
                    pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            h, w, _ = img_meta[i]['img_shape']
            if cut_mode is 'clamp':
                pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
                pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
                proposals_valid_list.append(pps_new.new_full(
                    (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            else:
                img_xyxy = pps_new.new_tensor([0, 0, w, h])
                iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
                proposals_valid = iof_in_img > 0.7
                proposals_valid_list.append(proposals_valid)

            proposal_list.append(pps_new.reshape(-1, 4))


    elif gen_mode == 'iou_based_random':
        num_gen = fine_proposal_cfg['gen_num_per_box']
        iou_thr = fine_proposal_cfg['iou_thr']
        proposal_list = []
        for i in range(len(img_meta)):
            proposal_list_single = hbb_proposalbox_aug(pseudo_boxes[i], num_gen, num_gen, iou_threshhold=iou_thr,
                                                       img_meta=img_meta[i])
            proposal_list.append(proposal_list_single)
    return proposal_list, proposals_valid_list


@DETECTORS.register_module()
class P2BNet(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
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
        super(P2BNet, self).__init__(
            backbone=backbone,
            neck=neck,
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

    def imbalance_proposal(self, valid_list, p):
        out_list = []
        for i, list in enumerate(valid_list):
            num_gt = len(p[i])
            list = list.reshape(num_gt, -1, 1)
            ll = []
            for j, c in enumerate(list):
                a = p[i][j]
                ll.append(torch.tensor(np.random.choice([1, 0], len(c), [a, 1 - a])).to(a.device))
            ll = torch.stack(ll).reshape(-1, 1)
            out_list.append(ll)
        return out_list

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

        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]

        for stage in range(self.num_stages):
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                if ann_weight is not None:
                    proposals_valid_list = self.imbalance_proposal(proposals_valid_list, ann_weight)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                neg_proposal_list, neg_weight_list = None, None
                # neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                #                                                             generate_proposals,
                #                                                             img_meta=img_metas)
                pseudo_boxes = generate_proposals
            # elif stage ==self.num_stages-1:
            #     generate_proposals, aug_generate_proposals = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
            #                                                                         img_meta=img_metas)
            #     neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
            #                                                                 generate_proposals,
            #                                                                 img_meta=img_metas)
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas,
                                                                                   stage=stage)
                if ann_weight is not None:
                    proposals_valid_list = self.imbalance_proposal(proposals_valid_list, ann_weight)
                # if stage == 1:
                #     neg_proposal_list, neg_weight_list = None, None
                # else:
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            roi_losses, pseudo_boxes, dynamic_weight,others = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_points,
                                                                                   gt_true_bboxes, gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   **kwargs)

            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        # if self.with_mask_head:
        # show(pseudo_boxes[0], gt_true_bboxes[0], img_metas[0])
        # roi_losses = self.roi_head.forward_train_mask(x, img_metas,
        #                                               pseudo_boxes_out,
        #                                               gt_true_bboxes, gt_labels,
        #                                               dynamic_weight,
        #                                               gt_bboxes_ignore, gt_masks,
        #                                               **kwargs)
        # for key, value in roi_losses.items():
        #     losses[f'{key}'] = value
        #
        if self.with_dense_head:

            losses_dense = self.dense_head.forward_train(x, img_metas, gt_bboxes,
                                                         pseudo_boxes_out, dynamic_weight,
                                                         gt_labels, gt_bboxes_ignore, gt_true_bboxes)
            for key, value in losses_dense.items():
                losses[f'pts_{key}'] = value

        elif self.with_mask_branch:
            cls_score, bbox_pred, centerness, param_pred = \
                self.bbox_head(x, self.mask_head.param_conv)
            bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (
                pseudo_boxes_out, gt_labels, img_metas)
            _, coors, level_inds, img_inds, gt_inds = self.bbox_head.loss(
                *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

            mask_feat = self.mask_branch(x)

            inputs = (cls_score, centerness, param_pred, coors, level_inds, img_inds, gt_inds)
            param_pred, coors, level_inds, img_inds, gt_inds = self.mask_head.training_sample(*inputs)
            mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)

            # self.show_imgs(mask_pred, level_inds, img_inds, gt_inds, img_metas, gt_labels, None, pseudo_boxes,
            #                gt_bboxes, mask_pred.shape[-2:])

            loss_mask = self.mask_head.loss(img, img_metas, mask_pred, gt_inds, pseudo_boxes_out,
                                            gt_masks, gt_labels)
            losses.update(loss_mask)

        # for i in range(len(img_metas)):
        #

        # losses.update(roi_losses)

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

            gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
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
                cls_scores
                ims = cv2.rectangle(ims, (gt_box[k, 0], gt_box[k, 1]), (gt_box[k, 2], gt_box[k, 3]),
                                    color=(0, 255, 0))
            # for i in range(len(gt_labels[i])):
            #     ims = cv2.rectangle(ims, (pos_box[i, 0], pos_box[i, 1]), (pos_box[i, 2], pos_box[i, 3]),
            #                          color=(0, 255, 0))

            for j in range(len(gt_labels[i])):
                m = min(gt_inds_)
                heatmap = cls_scores_[gt_inds_ - m == j]
                if len(heatmap) > 0:
                    heatmap = heatmap.sigmoid()
                    heatmap = heatmap.mean(dim=0).squeeze(0)
                    # heatmap = heatmap.permute(1, 0)
                    # heatmap[heatmap>0.1]=1
                    heatmap = np.array(heatmap.cpu().detach())
                    heatmapshow = None
                    # heatmap[0, 0] = 1
                    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)

                    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                    pad_h, pad_w, _ = img_meta['pad_shape']
                    im_h, im_w, _ = img_meta['img_shape']
                    heatmapshow = cv2.resize(heatmapshow, (w * 4, h * 4))

                    heatmapshow = heatmapshow[:im_h, :im_w, :]
                    img = cv2.imread(img_meta['filename'])
                    img = cv2.resize(img, (im_w, im_h))
                    img = cv2.rectangle(img, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                        color=(0, 255, 0))
                    img = cv2.rectangle(img, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                        color=(0, 255, 255))
                    heatmapshow = cv2.rectangle(heatmapshow, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                                color=(0, 255, 0))
                    heatmapshow = cv2.rectangle(heatmapshow, (pos_box[j, 0], pos_box[j, 1]),
                                                (pos_box[j, 2], pos_box[j, 3]),
                                                color=(0, 255, 255))

                    cv2.namedWindow("ims1", 0)
                    cv2.resizeWindow("ims1", 640, 480)
                    cv2.imshow('ims1', ims)
                    cv2.namedWindow("ht", 0)
                    cv2.resizeWindow("ht", im_w * 4, im_h * 4)
                    cv2.imshow('ht', heatmapshow)
                    cv2.namedWindow("im", 0)
                    cv2.resizeWindow("im", im_w * 4, im_h * 4)
                    cv2.imshow('im', img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()


def show(pseudo_box, gt_box, img_meta):
    import cv2
    num_gt = len(gt_box)
    ims = cv2.imread(img_meta['filename'])
    im_h, im_w, _ = img_meta['img_shape']
    img = cv2.resize(ims, (im_w, im_h))
    for j in range(len(gt_box)):
        gt_box=gt_box.int()
        img = cv2.rectangle(img, (int(gt_box[j, 0]), int(gt_box[j, 1])), (int(gt_box[j, 2]), int(gt_box[j, 3])),
                            color=(0, 255, 0))
        img = cv2.rectangle(img, (int(pseudo_box[j, 0]), int(pseudo_box[j, 1])), (int(pseudo_box[j, 2]), int(pseudo_box[j, 3])),
                            color=(0, 255, 255))
        cv2.namedWindow("ims1", 0)
        cv2.resizeWindow("ims1", 640, 480)
        cv2.imshow('ims1', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
