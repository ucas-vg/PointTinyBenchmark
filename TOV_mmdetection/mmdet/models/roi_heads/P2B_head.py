import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .cascade_roi_head import CascadeRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
import math
import copy
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_


@HEADS.register_module()
class P2BHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, bbox_roi_extractor, num_stages, bbox_head, top_k=7, with_atten=None, **kwargs):
        super(P2BHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk1 = top_k  # 7
        self.topk2 = top_k  # 7

        self.featmap_stride = bbox_roi_extractor.featmap_strides
        self.with_atten = with_atten


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
                      proposal_list_base,
                      proposals_list,
                      proposals_valid_list,
                      neg_proposal_list,
                      neg_weight_list,
                      gt_points,
                      gt_labels,
                      dynamic_weight,
                      gt_points_ignore=None,
                      gt_masks=None,
                      ):

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                    neg_proposal_list,
                                                    neg_weight_list,
                                                    gt_points, gt_labels, dynamic_weight,
                                                    img_metas, stage)

            losses.update(bbox_results['loss_instance_mil'])
        return losses, bbox_results['pseudo_boxes'], bbox_results['dynamic_weight']

    def atten_pts_and_ftmap(self, x, gt_bboxes, img_metas):
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
        out_feat = []
        for i in range(len(x)):
            out_feat_batch = []
            for bt in range(len(x[i])):
                ft = x[i][bt]
                ft = ft.reshape(1, ft.shape[0], -1)
                gt_pts = (gt_points[bt] / self.featmap_stride[i]).round().long()
                gt_pts[:, 0] = torch.clamp(gt_pts[:, 0], 0, x[i][bt].shape[-1] - 1)
                gt_pts[:, 1] = torch.clamp(gt_pts[:, 1], 0, x[i][bt].shape[-2] - 1)
                ft_pts = x[i][bt, None, :, gt_pts[:, 1], gt_pts[:, 0]]
                out = self.Pts_attention.forward(ft, ft_pts)
                out = out.reshape(*out.shape[:2], *x[i][bt].shape[-2:])
                out_feat_batch.append(out)
            out_feat.append(torch.cat(out_feat_batch))
        return out_feat

    def _bbox_forward_train(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                            neg_weight_list, gt_points,
                            gt_labels,
                            cascade_weight,
                            img_metas, stage):
        """Run forward function and calculate loss for box head in training."""

        rois = bbox2roi(proposals_list)
        bbox_results = self._bbox_forward(x, rois, gt_points, stage)
        num_instance = bbox_results['num_instance']
        gt_labels = torch.cat(gt_labels)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)

        if neg_proposal_list is not None:
            neg_rois = bbox2roi(neg_proposal_list)
            neg_bbox_results = self._bbox_forward(x, neg_rois, None, stage)  ######stage
            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
        else:
            neg_cls_scores = None
            neg_weights = None
        reg_box = bbox_results['bbox_pred']
        if reg_box is not None:
            boxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 4),
                                                          reg_box.reshape(-1, 4)).reshape(reg_box.shape)
        else:
            boxes_pred = None

        proposals_list_to_merge = proposals_list

        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_box(bbox_results,
                                                                                                  proposals_list_to_merge,
                                                                                                  proposals_valid_list,
                                                                                                  gt_labels,
                                                                                                  gt_points,
                                                                                                  img_metas, stage)
        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=dynamic_weight.sum(dim=-1))

        pseudo_boxes = torch.cat(pseudo_boxes)
        if stage == self.num_stages - 1:
            retrain_weights = None ##TO
        else:
            retrain_weights = None
        loss_instance_mil = self.bbox_head.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
                                                    proposals_valid_list,
                                                    neg_cls_scores, neg_weights,
                                                    boxes_pred, gt_labels,
                                                    torch.cat(proposal_list_base), label_weights=cascade_weight,
                                                    retrain_weights=retrain_weights)
        loss_instance_mil.update({"mean_ious": mean_ious[-1]})
        loss_instance_mil.update({"s": mean_ious[0]})
        loss_instance_mil.update({"m": mean_ious[1]})
        loss_instance_mil.update({"l": mean_ious[2]})
        loss_instance_mil.update({"h": mean_ious[3]})

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

    def _bbox_forward(self, x, rois, gt_points, stage):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)

        # positive sample
        if gt_points is not None:
            num_gt = torch.cat(gt_points).shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_points}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])

            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=num_gt)
            return bbox_results
        # megative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=None)
            return bbox_results

    def merge_box_single(self, cls_score, ins_score, dynamic_weight, gt_point, gt_label, proposals, img_metas, stage):
        if stage < self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'
        elif stage == self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'

        proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 4)
        h, w, c = img_metas['img_shape']
        num_gt, num_gen = proposals.shape[:2]
        # proposals = proposals.reshape(-1,4)
        if merge_mode == 'weighted_cls_topk':
            cls_score_, idx = cls_score.topk(k=self.topk2, dim=1)
            weight = cls_score_.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            boxes = (proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx] * weight).sum(dim=1)
            # print(weight.sum(dim=1))
            # print(boxes)
            return boxes, None, None

        if merge_mode == 'weighted_clsins_topk':
            if stage == 0:
                k = self.topk1
            else:
                k = self.topk2
            dynamic_weight_, idx = dynamic_weight.topk(k=k, dim=1)
            weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 4])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            boxes = (filtered_boxes * weight).sum(dim=1)
            h, w, _ = img_metas['img_shape']
            boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
            boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
            # print(weight.sum(dim=1))
            # print(boxes)
            filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   dynamic_weight=dynamic_weight_)

            return boxes, filtered_boxes, filtered_scores


    def merge_box(self, bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, img_metas, stage):
        cls_scores = bbox_results['cls_score']
        ins_scores = bbox_results['ins_score']
        num_instances = bbox_results['num_instance']
        # num_gt = len(gt_labels)

        # num_gt * num_box * num_class
        if stage < 1:
            cls_scores = cls_scores.softmax(dim=-1)
        else:
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
        cls_scores = torch.split(cls_scores, batch_gt)
        ins_scores = torch.split(ins_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
        if not isinstance(proposals_list, list):
            proposals_list = torch.split(proposals_list, batch_gt)
        stage_ = [stage for _ in range(len(cls_scores))]
        boxes, filtered_boxes, filtered_scores = multi_apply(self.merge_box_single, cls_scores, ins_scores,
                                                             dynamic_weight_list,
                                                             gt_bboxes,
                                                             gt_labels,
                                                             proposals_list,
                                                             img_metas, stage_)

        pseudo_boxes = torch.cat(boxes).detach()
        # mean_ious =torch.tensor(mean_ious).to(gt_point.device)
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

        if self.test_mean_iou and stage == 1:
            self.sum_iou += iou1.sum()
            self.sum_num += len(iou1)
            # time.sleep(0.01)  # 这里为了查看输出变化，实际使用不需要sleep
            print('\r', self.sum_iou / self.sum_num, end='', flush=True)

        pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
        return list(pseudo_boxes), mean_ious, list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()


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

    def simple_test(self,
                    stage,
                    x,
                    proposal_list,
                    proposals_valid_list,
                    gt_bboxes,
                    gt_labels,
                    gt_anns_id,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, pseudo_bboxes = self.simple_test_bboxes(
            x, img_metas, proposal_list, proposals_valid_list, gt_bboxes, gt_labels, gt_anns_id, stage, self.test_cfg,
            rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # pseudo_bboxes = [i[:, :4] for i in det_bboxes]
        return bbox_results, pseudo_bboxes

    def simple_test_bboxes(self,
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

        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_box(bbox_results,
                                                                                                  proposals,
                                                                                                  proposals_valid_list,
                                                                                                  torch.cat(gt_labels),
                                                                                                  gt_bboxes,
                                                                                                  img_metas, stage)
        pseudo_boxes_out = copy.deepcopy(pseudo_boxes)

        det_bboxes, det_labels = self.pseudobox_to_result(pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id,
                                                          scale_factors, rescale)

        return det_bboxes, det_labels, pseudo_boxes_out

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

