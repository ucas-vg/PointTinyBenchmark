import copy

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import cv2
import numpy as np
from mmdet.models.builder import build_head
from mmdet.core.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core.point.p2b_utils.box_sampler import CBP_proposals_from_cfg, gen_negative_proposals, \
    PBR_proposals_from_cfg, random_shake_sample, random_scale_sample
import torch.nn.functional as F


def enlarge_box(pseudo_box):
    list = []
    for pps in pseudo_box:
        xywh = bbox_xyxy_to_cxcywh(pps)
        xywh[:, 2] = xywh[:, 2] * 1.1
        xywh[:, 3] = xywh[:, 3] * 1.1
        pps_new = bbox_cxcywh_to_xyxy(xywh)
        list.append(pps_new)
    return list


def shake_box(pseudo_box, direct, rate=0.3):
    list = []
    for pps in pseudo_box:
        xywh = bbox_xyxy_to_cxcywh(pps)
        if direct == 'left':
            xywh[:, 0] = xywh[:, 0] - xywh[:, 2] * rate
        elif direct == 'right':
            xywh[:, 0] = xywh[:, 0] + xywh[:, 2] * rate
        elif direct == 'up':
            xywh[:, 1] = xywh[:, 1] - xywh[:, 3] * rate
        elif direct == 'down':
            xywh[:, 1] = xywh[:, 1] + xywh[:, 3] * rate
        pps_new = bbox_cxcywh_to_xyxy(xywh)
        list.append(pps_new)
    return list


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
                cascade_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                if self.roi_head.stage_modes[stage] == 'PBR':
                    neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, base_proposal_cfg,
                                                                                generate_proposals, img_meta=img_metas)
                else:
                    neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = None

                # import cv2
                # import matplotlib.pyplot as plt
                # import numpy as np
                # img_path = img_metas[0]['filename']
                # img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道，因为cv2读取是BGR，而matplotlib展示需要RGB
                #
                # # 将边界框坐标还原到原始图像尺寸（考虑可能存在的缩放等变换）
                # scale_factor = img_metas[0]['scale_factor']
                # generate_proposals[0][:, 0] /= scale_factor[0]
                # generate_proposals[0][:, 1] /= scale_factor[1]
                # generate_proposals[0][:, 2] /= scale_factor[0]
                # generate_proposals[0][:, 3] /= scale_factor[1]
                #
                # # 绘制边界框
                # for proposal in generate_proposals:
                #     x1, y1, x2, y2 = proposal.astype(int)
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 这里以绿色绘制边界框，线宽为2
                #
                # # 使用matplotlib展示图像
                # plt.imshow(img)
                # plt.axis('off')
                # plt.show()









            else:
                if 'reg_proposal_boxes' in bbox_results:
                    generate_proposals = bbox_results['reg_proposal_boxes']
                    proposals_valid_list = [i.new_full(i.shape[:1], 1) for i in generate_proposals]
                else:
                    generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                      img_meta=img_metas,
                                                                                      stage=stage)

                    with_distill = self.roi_head.bbox_head.with_ori_distill if hasattr(self.roi_head.bbox_head,
                                                                                       'with_ori_distill') else False
                    if with_distill:
                        label_weights = cascade_weight
                        loss_distill, generate_proposals, proposals_valid_list = self.roi_head.forward_train_distill(x,
                                                                                                                     img_metas,
                                                                                                                     generate_proposals,
                                                                                                                     proposals_valid_list,
                                                                                                                     pseudo_boxes,
                                                                                                                     gt_true_bboxes,
                                                                                                                     gt_labels,
                                                                                                                     label_weights,
                                                                                                                     stage_mode='ori')
                        for key, value in loss_distill.items():
                            losses[f'ori_{key}'] = value
                    if ann_weight is not None:
                        proposals_valid_list = self.imbalance_proposal(proposals_valid_list, ann_weight)
                    neg_proposal_list, neg_weight_list = gen_negative_proposals(pseudo_boxes, fine_proposal_cfg,
                                                                                generate_proposals,
                                                                                img_meta=img_metas)
            roi_losses, bbox_results = self.roi_head.forward_train(stage, x, img_metas, pseudo_boxes,
                                                                   generate_proposals,
                                                                   proposals_valid_list,
                                                                   neg_proposal_list,
                                                                   neg_weight_list,
                                                                   gt_points,
                                                                   gt_true_bboxes, gt_labels,
                                                                   cascade_weight,
                                                                   gt_bboxes_ignore, gt_masks,
                                                                   **kwargs)
            pseudo_boxes, cascade_weight, others = bbox_results['pseudo_boxes'], bbox_results['dynamic_weight'], \
                                                   bbox_results['others']

            # if stage == 0:
            #     pseudo_boxes_out = pseudo_boxes
            #     dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value

        if self.with_mask_branch:
            # import numpy as np
            # import cv2
            #
            # cv2.namedWindow("gt", 0)
            # cv2.moveWindow("gt", 3200, 100)
            # filename = img_metas[0]['filename']
            # igs = cv2.imread(filename)
            # cv2.imshow('gt', igs)
            # if cv2.waitKey(0):
            #     cv2.destroyAllWindows()
            # pseudo_boxes=gt_true_bboxes
            # pseudo_boxes = enlarge_box(pseudo_boxes)
            mask_feat = self.mask_branch(x[1:])
            num_level = len(x[1:])
            inputs = (pseudo_boxes, gt_labels, num_level, others, cascade_weight)

            param_pred, coors, level_inds, img_inds, pseudo_boxes, cascade_weight = self.mask_head.training_sample(
                *inputs)
            if self.mask_head.with_class_embedding:
                mask_pred_list = []
                for param in param_pred:
                    mask_pred = self.mask_head(mask_feat, param, coors, level_inds, img_inds)
                    mask_pred_list.append(mask_pred)
                mask_pred = torch.cat(mask_pred_list, dim=1)
            else:
                mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)
            # mask_pred=
            # import numpy as np
            # import cv2
            #
            # cv2.namedWindow("ims1", 0)
            # cv2.namedWindow("gt", 0)
            # cv2.moveWindow("ims1", 2500, 100)
            # cv2.moveWindow("gt" , 3200, 100)
            # filename = img_metas[0]['filename']
            # igs = cv2.imread(filename)
            # a = mask_pred.sigmoid()[0, 0]
            # cv2.imshow('ims1', np.array(a.detach().cpu()))
            # cv2.imshow('gt', igs)
            # if cv2.waitKey(0):
            #     cv2.destroyAllWindows()
            # import numpy as np
            # import cv2
            # cv2.namedWindow("ims1", 0)
            # cv2.namedWindow("gt", 0)
            # cv2.moveWindow("ims1", 2500, 100)
            # cv2.moveWindow("gt", 3200, 100)
            # filename = img_metas[0]['filename']
            # igs = cv2.imread(filename)
            # a = mask_pred.sigmoid()[0, 0]
            # a = torch.stack([a, a, a], dim=2)
            # a = np.array(a.detach().cpu())
            # a = cv2.resize(a, (a.shape[1] * 4, a.shape[0] * 4))
            # # pps=torch.cat(generate_proposals).reshape(len(mask_pred), -1, 4)[0]
            # bbx = coors[0][0]
            # # for bx in pps:
            # # # bx = coors[0][0]
            # #     a = cv2.rectangle(img=a, pt1=(int(bx[0]), int(bx[1])), pt2=(int(bx[2]), int(bx[3])), color=[0,0, 1.0],
            # #                       thickness=10)
            # a = cv2.rectangle(img=a, pt1=(int(bbx[0] - bbx[2] / 2), int(bbx[1] - bbx[3] / 2)),
            #                   pt2=(int(bbx[0] + bbx[2] / 2), int(bbx[1] + bbx[3] / 2)), color=[0, 0, 1.0], thickness=10)
            # gt = gt_true_bboxes[0][0]
            # a = cv2.rectangle(img=a, pt1=(int(gt[0]), int(gt[1])), pt2=(int(gt[2]), int(gt[3])), color=[0, 1.0, 0],
            #                   thickness=10)
            # cv2.imshow('ims1', a)
            # cv2.imshow('gt', igs)
            # if cv2.waitKey(0):
            #     cv2.destroyAllWindows()
            # show(pseudo_boxes,gt)
            shake_list = []
            shake_times = 5
            for i in range(shake_times):
                pseudo_boxes_shake = random_scale_sample(pseudo_boxes, cascade_weight)
                shake_list.append(pseudo_boxes_shake)
            loss_mask = self.mask_head.loss(img, img_metas, mask_pred, torch.arange(len(mask_pred)), pseudo_boxes,
                                            gt_true_bboxes,
                                            gt_masks, gt_labels, cascade_weight, pseudo_boxes_shake=None)
            for key, value in loss_mask.items():
                losses[f'{key}'] = value

            # loss_mask_shake = self.mask_head.loss(img, img_metas, mask_pred, torch.arange(len(mask_pred)),
            #                                       pseudo_boxes_shake,
            #                                       gt_true_bboxes,
            #                                       gt_masks, gt_labels, cascade_weight, SHAKE=True)
            # for key, value in loss_mask_shake.items():
            #     losses[f'shake_{key}'] = value * 0.5
            losses.update(loss_mask)

            mask_pred_init = self.mask_head.get_mask(mask_pred[:, 0, None].detach(), others, img_metas)

            masks_mean_iou, boxes_mean_iou, pseudo_boxes, mask_pred_list = self.mask_head.mask_mean_iou(mask_pred_init,
                                                                                                        gt_masks,
                                                                                                        gt_true_bboxes,
                                                                                                        pseudo_boxes,
                                                                                                        img_metas)
            if masks_mean_iou is not None:
                losses['masks_mean_iou'] = masks_mean_iou
            losses['boxes_mean_iou'] = boxes_mean_iou
            if self.mask_head.enlarge_rate:
                refined_mask_pred = self.mask_head.refiner_forward(mask_pred[:, 1, None].detach(), mask_feat,
                                                                   param_pred, img_metas,
                                                                   img_inds)
                refine_loss = self.mask_head.refiner_loss(refined_mask_pred, (mask_pred_init > 0.5).type(torch.int8),
                                                          cascade_weight)
                losses.update(refine_loss)
                mask_pred_refine = self.mask_head.refiner_forward(mask_pred[:, 0, None], mask_feat, param_pred,
                                                                  img_metas,
                                                                  img_inds)
                mask_pred_refine = self.mask_head.get_mask(mask_pred_refine.detach(), others, img_metas)
                masks_mean_iou, boxes_mean_iou, pseudo_boxes, mask_pred_list = self.mask_head.mask_mean_iou(
                    mask_pred_refine,
                    gt_masks,
                    gt_true_bboxes,
                    pseudo_boxes,
                    img_metas)
                losses['refine_masks_mean_iou'] = masks_mean_iou
                losses['refine_boxes_mean_iou'] = boxes_mean_iou
            if self.mask_head.mil_proj:
                mask_pred_refine = self.mask_head.get_mask(mask_pred[:, 1, None].detach(), others, img_metas)
                masks_mean_iou, boxes_mean_iou, pseudo_boxes, mask_pred_list = self.mask_head.mask_mean_iou(
                    mask_pred_refine,
                    gt_masks,
                    gt_true_bboxes,
                    pseudo_boxes,
                    img_metas)
                losses['refine_masks_mean_iou'] = masks_mean_iou
                losses['refine_boxes_mean_iou'] = boxes_mean_iou

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

    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""
    #
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     if proposals is None:
    #         proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals
    #
    #     return self.det_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)
    #
    def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id, gt_true_bboxes, gt_labels,
                    gt_bboxes_ignore=None, gt_masks=None, proposals=None, rescale=False):
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
                generate_proposals, proposals_valid_list = CBP_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
            else:
                if 'reg_proposal_boxes' in others:
                    generate_proposals = others['reg_proposal_boxes']
                    proposals_valid_list = [i.new_full(i.shape[:1], 1) for i in generate_proposals]
                else:
                    generate_proposals, proposals_valid_list = PBR_proposals_from_cfg(pseudo_bboxes, fine_proposal_cfg,
                                                                                      img_meta=img_metas, stage=stage)
                    with_distill = self.roi_head.bbox_head.with_ori_distill if hasattr(self.roi_head.bbox_head,
                                                                                       'with_ori_distill') else False
                    if with_distill:
                        label_weights = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                        generate_proposals, proposals_valid_list = self.roi_head.forward_test_distill(x,
                                                                                                      img_metas,
                                                                                                      generate_proposals,
                                                                                                      proposals_valid_list,
                                                                                                      gt_bboxes,
                                                                                                      gt_true_bboxes,
                                                                                                      gt_labels,
                                                                                                      label_weights,
                                                                                                      stage_mode='ori')
            test_result, pseudo_bboxes, others = self.roi_head.simple_test_pseudo(stage,
                                                                                  x, generate_proposals,
                                                                                  proposals_valid_list,
                                                                                  gt_true_bboxes, gt_labels,
                                                                                  gt_anns_id,
                                                                                  img_metas,
                                                                                  rescale=rescale)

        if not self.with_mask_branch:
            return test_result
        else:
            x = x[1:]
            mask_feat = self.mask_branch(x)
            num_level = len(x)
            inputs = (pseudo_bboxes, gt_labels, num_level, others)

            param_pred, coors, level_inds, img_inds, pseudo_boxes, cascade_weight = self.mask_head.training_sample(
                *inputs,
                test=True)
            # mask_pred = self.mask_head(mask_feat, param_pred, coors, level_inds, img_inds)
            bbox_results, mask_result = self.mask_head.simple_test_mask(mask_feat, gt_labels, param_pred, coors,
                                                                        level_inds, img_inds, img_metas,
                                                                        self.roi_head.bbox_head.num_classes,
                                                                        others['dynamic_weight'], gt_anns_id, others,
                                                                        gt_true_bboxes, gt_masks,
                                                                        rescale=rescale)

            return list(zip(test_result, mask_result))

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

def show(pseudo_box, gt_box, img_meta):
    for i in range(len(gt_true_bboxes)):
        bboxes = gt_true_bboxes[i]
        mask_p = mask_pred[i]
        recta = rectangle[i]
        import numpy as np
        import cv2
        for j in range(len(mask_p)):
            cv2.namedWindow("ims1", 0)
            cv2.namedWindow("gt", 0)
            filename = img_metas[i]['filename']
            igs = cv2.imread(filename)
            map = mask_p[j, 0]
            bbox = bboxes[j]
            rect = recta[j]
            img_shape = img_metas[i]['img_shape']
            igs = cv2.resize(igs, (img_shape[1], img_shape[0]))
            # igs = cv2.resize(igs, None, fx=self.mask_head.out_stride, fy=self.mask_head.out_stride)
            map = cv2.resize(np.array(map.detach().cpu()), None, fx=self.mask_head.out_stride,
                             fy=self.mask_head.out_stride)
            map=map[:img_shape[0],:img_shape[1]]
            map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
            map = cv2.rectangle(map, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),
                                color=(0, 0, 255),
                                thickness=5)
            igs = cv2.rectangle(igs, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                color=(0, 255, 0),
                                thickness=5)
            cv2.imshow('ims1', map)
            cv2.imshow('gt', igs)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()
