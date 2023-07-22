import copy

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import cv2
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import build_head
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.point.p2b_utils.box_sampler import CBP_proposals_from_cfg, gen_negative_proposals, \
    PBR_proposals_from_cfg
import torch.nn.functional as F


@DETECTORS.register_module()
class P2Seg(TwoStageDetector):
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
                 rpn_head=None,
                 det_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(P2Seg, self).__init__(
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
        if mask_head is not None:
            self.with_mask_head = True
            self.mask_head = build_head(mask_head)

        if det_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            self.with_det_head = True
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            det_head.update(train_cfg=rcnn_train_cfg)
            det_head.update(test_cfg=test_cfg.rcnn)
            det_head.pretrained = pretrained
            self.det_head = build_head(det_head)

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
            if stage == 0:
                generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                if ann_weight is not None:
                    proposals_valid_list = self.imbalance_proposal(proposals_valid_list, ann_weight)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = generate_proposals

            else:
                generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                  img_meta=img_metas,
                                                                                  stage=stage)
                if ann_weight is not None:
                    proposals_valid_list = self.imbalance_proposal(proposals_valid_list, ann_weight)
                neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_boxes, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            roi_losses, pseudo_boxes, dynamic_weight, others = self.roi_head.forward_train(stage, x, img_metas,
                                                                                           pseudo_boxes,
                                                                                           generate_proposals,
                                                                                           proposals_valid_list,
                                                                                           neg_proposal_list,
                                                                                           neg_weight_list,
                                                                                           gt_points,
                                                                                           gt_true_bboxes, gt_labels,
                                                                                           dynamic_weight,
                                                                                           gt_bboxes_ignore, gt_masks,
                                                                                           **kwargs)

            # if stage == 0:
            #     pseudo_boxes_out = pseudo_boxes
            #     dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        if self.with_mask_branch:
            mask_feat = self.mask_branch(x[1:])
            num_level = len(x[1:])
            inputs = (pseudo_boxes, gt_labels, num_level, others)

            param_pred, coors, level_inds, img_inds = self.mask_head.training_sample(*inputs)
            mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)

            # import numpy as np
            # import cv2
            #
            # cv2.namedWindow("ims1", 0)
            # cv2.namedWindow("gt", 0)
            # filename = img_metas[0]['filename']
            # igs = cv2.imread(filename)
            # a = mask_pred.sigmoid()[0, 0]
            # cv2.imshow('ims1', np.array(a.detach().cpu()))
            # cv2.imshow('gt', igs)
            # if cv2.waitKey(0):
            #     cv2.destroyAllWindows()
            loss_mask = self.mask_head.loss(img, img_metas, mask_pred, torch.arange(len(mask_pred)), pseudo_boxes,
                                            gt_masks, gt_labels, dynamic_weight)
            masks_mean_iou, boxes_mean_iou, pseudo_boxes, mask_pred_list = self.mask_mean_iou(mask_pred.detach(), gt_masks,
                                                                                           gt_true_bboxes,
                                                                                           pseudo_boxes, img_metas)
            mask_pred = torch.split(mask_pred.sigmoid(), [len(i) for i in gt_true_bboxes])

            losses.update(loss_mask)
            losses['masks_mean_iou'] = masks_mean_iou
            losses['boxes_mean_iou'] = boxes_mean_iou

        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         pseudo_boxes,
        #         gt_labels=None,
        #         ann_weight=None,  # dynamic_weight,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg)
        #     losses.update(rpn_losses)
        # if self.with_det_head:
        #     for i in range(len(gt_masks)):
        #         gt_masks[i].masks = np.array(mask_pred_list[i].detach().cpu())
        #     det_losses = self.det_head.forward_train(x, img_metas, proposal_list, pseudo_boxes,
        #                                              gt_labels, ann_weight, gt_bboxes_ignore,
        #                                              gt_masks, **kwargs)
        #     for key, value in det_losses.items():
        #         losses[f'det_{key}'] = value
        return losses

    def aligned_bilinear(self, tensor, factor):
        assert tensor.dim() == 4
        assert factor >= 1
        assert isinstance(factor, int)

        if factor == 1:
            return tensor

        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(
            tensor, size=(oh, ow),
            mode='bilinear',
            align_corners=True
        )
        tensor = F.pad(
            tensor, pad=(factor // 2, 0, factor // 2, 0),
            mode="replicate"
        )
        return tensor[:, :, :oh - 1, :ow - 1]

    def mask_mean_iou(self, mask_pred, gt_masks, gt_bboxes, pseudo_boxes, img_metas):

        mask_pred = mask_pred.sigmoid().type(torch.float)
        stride = self.mask_head.out_stride
        mask_pred = self.aligned_bilinear(mask_pred, stride).squeeze(1)
        mask_pred = (mask_pred > 0.1).type(torch.int8)
        pseudo_bboxes = self.masks_to_boxes(mask_pred)
        gt_bboxes = torch.cat(gt_bboxes)
        iou_box = bbox_overlaps(pseudo_bboxes, gt_bboxes, is_aligned=True)
        gt_masks = [i.masks for i in gt_masks]
        batch_split = [len(i) for i in gt_masks]
        pseudo_bboxes = pseudo_bboxes.split(batch_split)
        # batch_split = self.batch_split_np(batch_split)
        mask_pred_list = mask_pred.split(batch_split)
        iou_sum, num = 0, 0
        for i, gt_mask in enumerate(gt_masks):
            gt_mask = torch.tensor(gt_mask).to(mask_pred_list[0].device)
            mask = mask_pred_list[i]
            shape = gt_mask.shape
            mask = mask[:, :shape[1], :shape[2]]
            intersection = ((mask + gt_mask) == 2).reshape(shape[0], -1).sum(1)
            union = ((mask + gt_mask) >= 1).reshape(shape[0], -1).sum(1)
            IoU = intersection / union
            iou_sum += IoU.sum()
            num += shape[0]

        return iou_sum / num, iou_box.mean(), pseudo_bboxes,mask_pred_list

    def batch_split_np(self, batch_split):
        a, len = [], 0
        for l in batch_split:
            len += l
            a.append(len)
        a.pop(-1)
        return a

    def masks_to_boxes(self, masks):
        """Compute the bounding boxes around the provided masks

        The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

        Returns a [N, 4] tensors, with the boxes in xyxy format
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)

        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float).to(masks.device)
        x = torch.arange(0, w, dtype=torch.float).to(masks.device)
        y, x = torch.meshgrid(y, x)

        x_mask = (masks * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        y_mask = (masks * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        return torch.stack([x_min, y_min, x_max, y_max], 1)

    def find_rect(self, mask_pred):
        mask_pred = torch.tensor(mask_pred)
        for i in range(mask_pred):
            mask_pred[i].nonzero()
        for mask in mask_pred:
            mask = mask.astype(np.uint8)[:, :, None]
            img_bin, contours, hierarchy = cv2.findContours(mask,
                                                            cv2.RETR_LIST,
                                                            cv2.CHAIN_APPROX_SIMPLE)
        return


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.det_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    #
    # def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id, gt_true_bboxes, gt_labels,
    #                 gt_bboxes_ignore=None, proposals=None, rescale=False):
    #     """Test without augmentation."""
    #     base_proposal_cfg = self.train_cfg.get('base_proposal',
    #                                            self.test_cfg.rpn)
    #     fine_proposal_cfg = self.train_cfg.get('fine_proposal',
    #                                            self.test_cfg.rpn)
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     for stage in range(self.num_stages):
    #
    #         gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
    #         if stage == 0:
    #             generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
    #                                                                               img_meta=img_metas)
    #         else:
    #             generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_bboxes, fine_proposal_cfg,
    #                                                                               img_meta=img_metas, stage=stage)
    #
    #         det_bboxes, det_labels, pseudo_bboxes, others = self.roi_head.simple_test_bboxes(x, img_metas,
    #                                                                                          generate_proposals,
    #                                                                                          proposals_valid_list,
    #                                                                                          gt_bboxes, gt_labels,
    #                                                                                          gt_anns_id, stage,
    #                                                                                          rcnn_test_cfg=None,
    #                                                                                          rescale=rescale)
    #         x = x[1:]
    #         mask_feat = self.mask_branch(x)
    #         num_level = len(x)
    #         inputs = (pseudo_bboxes, gt_labels, num_level, others)
    #
    #         param_pred, coors, level_inds, img_inds = self.mask_head.training_sample(*inputs)
    #
    #         mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)
    #
    #     return mask_pred

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

# def show(pseudo_box, gt_box, img_meta):
# for i in range(len(gt_true_bboxes)):
#     bboxes = gt_true_bboxes[i]
#     mask_p = mask_pred[i]
#     recta = rectangle[i]
#     import numpy as np
#     import cv2
#     for j in range(len(mask_p)):
#         cv2.namedWindow("ims1", 0)
#         cv2.namedWindow("gt", 0)
#         filename = img_metas[i]['filename']
#         igs = cv2.imread(filename)
#         map = mask_p[j, 0]
#         bbox = bboxes[j]
#         rect = recta[j]
#         img_shape = img_metas[i]['img_shape']
#         igs = cv2.resize(igs, (img_shape[1], img_shape[0]))
#         # igs = cv2.resize(igs, None, fx=self.mask_head.out_stride, fy=self.mask_head.out_stride)
#         map = cv2.resize(np.array(map.detach().cpu()), None, fx=self.mask_head.out_stride,
#                          fy=self.mask_head.out_stride)
#         map=map[:img_shape[0],:img_shape[1]]
#         map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
#         map = cv2.rectangle(map, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),
#                             color=(0, 0, 255),
#                             thickness=5)
#         igs = cv2.rectangle(igs, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
#                             color=(0, 255, 0),
#                             thickness=5)
#         cv2.imshow('ims1', map)
#         cv2.imshow('gt', igs)
#         if cv2.waitKey(0):
#             cv2.destroyAllWindows()
