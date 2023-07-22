from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np


def CBP_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    if 'cut_mode' in proposal_cfg:
        cut_mode = proposal_cfg['cut_mode']
    else:
        cut_mode = 'symmetry'
    base_proposal_list = []
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
        if cut_mode == 'clamp':
            base_proposals[..., 0:4:2] = torch.clamp(base_proposals[..., 0:4:2], 0, img_w)
            base_proposals[..., 1:4:2] = torch.clamp(base_proposals[..., 1:4:2], 0, img_h)
            proposals_valid_list.append(proposals_valid)
        if cut_mode == 'symmetry':
            proposals_valid_list.append(proposals_valid)
        elif cut_mode == 'ignore':
            img_xyxy = base_proposals.new_tensor([0, 0, img_w, img_h])
            iof_in_img = bbox_overlaps(base_proposals, img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
        elif cut_mode is None:
            proposals_valid_list.append(proposals_valid)
        base_proposal_list.append(base_proposals)

    return base_proposal_list, proposals_valid_list


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    iou_thr = proposal_cfg['iou_thr']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_bboxes = torch.stack([x1, y1, x2, y2], dim=1).to(gt_points[0].device)
        iou = bbox_overlaps(neg_bboxes, pos_box)
        neg_weight = ((iou < iou_thr).sum(dim=1) == iou.shape[1])

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    return neg_proposal_list, neg_weight_list


def PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    if isinstance(fine_proposal_cfg['base_ratios'], tuple):
        base_ratios = fine_proposal_cfg['base_ratios'][stage]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage]
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

    return proposal_list, proposals_valid_list


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
