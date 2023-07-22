import torch
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import multi_apply
import torch.nn.functional as F


def merge_box_single(cls_score, ins_score, dynamic_weight, gt_point, gt_label, proposals, feat, img_metas, stage_mode,
                     topk, flag=None):
    if stage_mode == 'CBP':
        merge_mode = 'weighted_clsins_topk'
    elif stage_mode == 'PBR':
        merge_mode = 'weighted_cls_topk' if flag == 'iou_pred' else 'weighted_clsins_topk'  ######### changed

    proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 4)
    h, w, c = img_metas['img_shape']
    num_gt, num_gen = proposals.shape[:2]
    # proposals = proposals.reshape(-1,4)
    if merge_mode == 'weighted_cls_topk':
        cls_score_, idx = cls_score.topk(k=topk, dim=1)
        # weight = cls_score_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = cls_score_ / cls_score_.sum(dim=1, keepdim=True) + 1e-8
        boxes = (proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx] * weight[:, :, None]).sum(dim=1)
        # print(weight.sum(dim=1))
        # print(boxes)
        if feat is not None:
            filtered_feat = feat[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            feat = (weight[:, :, 0] * filtered_feat).sum(1)
        else:
            feat = None
        return boxes, None, None, feat

    if merge_mode == 'weighted_clsins_topk':
        dynamic_weight_, idx = dynamic_weight.topk(k=topk, dim=1)
        # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = dynamic_weight_ / dynamic_weight_.sum(dim=1, keepdim=True) + 1e-8
        filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        if feat is not None:
            filtered_feat = feat[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            feat = None
        h, w, _ = img_metas['img_shape']
        boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
        boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
        # print(weight.sum(dim=1))
        # print(boxes)
        filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                               ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                               dynamic_weight=dynamic_weight_)

        return boxes, filtered_boxes, filtered_scores, feat

    if merge_mode == 'weighted_clsins_max_iou':
        dynamic_weight_, idx = dynamic_weight.topk(k=topk, dim=1)
        # weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
        # weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        weight = dynamic_weight_ / dynamic_weight_.sum(dim=1, keepdim=True) + 1e-8
        filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
        boxes = (filtered_boxes * weight[:, :, None]).sum(dim=1)
        if feat is not None:
            filtered_feat = feat[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            feat = (weight[:, :, None] * filtered_feat).sum(1)
        else:
            feat = None
        h, w, _ = img_metas['img_shape']
        boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
        boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
        # print(weight.sum(dim=1))
        # print(boxes)
        filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                               ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                               dynamic_weight=dynamic_weight_)

        return boxes, filtered_boxes, filtered_scores, feat


def merge_box(bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, img_metas, stage_mode, topk,
              proposal_list_base=None,flag=None):
    cls_scores = bbox_results['cls_score']
    ins_scores = bbox_results['ins_score']
    num_instances = bbox_results['num_instance']
    num_gt = len(gt_labels)

    # num_gt * num_box * num_class
    if stage_mode == 'CBP':
        cls_scores = cls_scores.softmax(dim=-1)
    elif stage_mode == 'PBR':
        cls_scores = cls_scores.sigmoid()
    ins_scores = ins_scores.softmax(dim=-2) * proposals_valid_list
    ins_scores = F.normalize(ins_scores, dim=1, p=1)
    cls_scores = cls_scores * proposals_valid_list
    dynamic_weight = (cls_scores * ins_scores)
    dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
    cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
    ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]
    # split batch
    batch_gt = [len(b) for b in gt_bboxes]
    if 'iou_score'  in bbox_results:
        if bbox_results['iou_score'] is not None:
            iou_scores = bbox_results['iou_score'].squeeze(-1)
            mean, std = 0.5, 0.5
            iou_scores = iou_scores * std + mean

            iou_scores = (iou_scores - iou_scores.min(1, keepdim=True)[0]) / (
                    iou_scores.max(1, keepdim=True)[0] - iou_scores.min(1, keepdim=True)[0])
            flag = 'iou_pred'
            # if bbox_results['obj_score'] is not None:
            #     obj_scores = bbox_results['obj_score']
            #     obj_scores = iou_scores * obj_scores.sigmoid()
            #     if obj_scores.shape[1] != 1:
            #         obj_scores = obj_scores.reshape(*cls_scores.shape, -1)[:, :, 0]
            #
            # else:
            #     obj_scores = obj_scores.reshape(cls_scores.shape).sigmoid()
            # obj_scores = obj_scores.sigmoid()
            cls_scores = cls_scores * iou_scores
            dynamic_weight = dynamic_weight * iou_scores

    if bbox_results['others']:
        feat = bbox_results['others']['x_feat']
        feat = feat.reshape(cls_scores.shape[0], -1, feat.shape[-1])
        feat = torch.split(feat, batch_gt)
    else:
        feat = [None for _ in range(len(batch_gt))]
    cls_scores = torch.split(cls_scores, batch_gt)
    ins_scores = torch.split(ins_scores, batch_gt)
    gt_labels = torch.split(gt_labels, batch_gt)
    dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
    if not isinstance(proposals_list, list):
        proposals_list = torch.split(proposals_list, batch_gt)
    stage_mode_ = [stage_mode for _ in range(len(cls_scores))]
    topk = [topk for _ in range(len(cls_scores))]
    boxes, filtered_boxes, filtered_scores, feat = multi_apply(merge_box_single, cls_scores, ins_scores,
                                                               dynamic_weight_list,
                                                               gt_bboxes,
                                                               gt_labels,
                                                               proposals_list, feat,
                                                               img_metas, stage_mode_, topk, flag=flag)
    if bbox_results['others']:
        bbox_results['others']['x_feat'] = feat
    pseudo_boxes = torch.cat(boxes).detach()
    # mean_ious =torch.tensor(mean_ious).to(gt_point.device)

    ## condition
    # pseudo_boxes1 = pseudo_boxes * (dynamic_weight.sum(-1,keepdim=True) >0.2)+ torch.cat( proposal_list_base) * (dynamic_weight.sum(-1,keepdim=True)<0.2)

    iou1 = bbox_overlaps(pseudo_boxes, torch.cat(gt_bboxes), is_aligned=True)

    ### scale mean iou
    gt_xywh = bbox_xyxy_to_cxcywh(torch.cat(gt_bboxes))
    scale = gt_xywh[:, 2] * gt_xywh[:, 3]
    mean_iou_s = iou1[scale < 32 ** 2].sum() / (len(iou1[scale < 32 ** 2]) + 1e-5)
    mean_iou_m = iou1[(scale > 32 ** 2) * (scale < 64 ** 2)].sum() / (len(
        iou1[(scale > 32 ** 2) * (scale < 64 ** 2)]) + 1e-5)
    mean_iou_l = iou1[(scale > 64 ** 2) * (scale < 128 ** 2)].sum() / (len(
        iou1[(scale > 64 ** 2) * (scale < 128 ** 2)]) + 1e-5)
    mean_iou_h = iou1[scale > 128 ** 2].sum() / (len(iou1[scale > 128 ** 2]) + 1e-5)

    mean_ious_all = iou1.mean()
    mean_ious = [mean_iou_s, mean_iou_m, mean_iou_l, mean_iou_h, mean_ious_all]
    #
    # if self.test_mean_iou and stage == 1:
    #     self.sum_iou += iou1.sum()
    #     self.sum_num += len(iou1)
    #     # time.sleep(0.01)  # 这里为了查看输出变化，实际使用不需要sleep
    #     print('\r', self.sum_iou / self.sum_num, end='', flush=True)

    pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
    return list(pseudo_boxes), mean_ious, list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()
