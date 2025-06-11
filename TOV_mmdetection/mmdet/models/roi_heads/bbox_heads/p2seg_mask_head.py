from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from timm.models.vision_transformer import Block

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit
from mmdet.models.losses import accuracy


@HEADS.register_module()
class P2SegMaskHead(BaseModule):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_mil=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25,
                     loss_type='binary_cross_entropy'),  # weight
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(P2SegMaskHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_mil = build_loss(loss_mil)

        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_cls = build_conv_layer(self.predictor_cfg,
                                         logits_in_channel, out_channels, 1)
        self.conv_ins = build_conv_layer(self.predictor_cfg,
                                         logits_in_channel, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        depth = 4
        self.blocks = nn.Sequential(*[
            Block(
                dim=256, num_heads=8, mlp_ratio=4., qkv_bias=True)
            for i in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.in_channels))
        self.pos_embed = nn.Parameter(torch.randn(1, roi_feat_size * roi_feat_size + 1, self.in_channels) * .02)

        self.fc_cls = nn.Linear(self.in_channels, num_classes)

    def init_weights(self):
        super(P2SegMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_cls, self.conv_ins, self.fc_cls]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        x_patch = x.reshape(x.shape[0], self.in_channels, -1).permute(0, 2, 1)
        x_patch = torch.cat((self.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        x_in = x_patch + self.pos_embed
        output = self.blocks((x_in, []))
        x, attn = output[0], output[1]
        attn_map = self.attns_project_to_feature(attn)
        x_cls = x[:, 0]
        # for conv in self.convs:
        #     x = conv(x)
        # mask_cls_pred = self.conv_cls(x)
        # mask_ins_pred = self.conv_ins(x)
        cls_score = self.fc_cls(x_cls)
        # cls_score=cls_score.softmax(dim=-1)
        return cls_score, None, attn_map

    def attns_project_to_feature(self, attns_maps):
        #         assert len(attns_maps[1]) == 1
        # [block_num], B, H, all_num, all_num
        attns_maps = torch.stack(attns_maps)
        # block_num, B, H, all_num, all_num
        attns_maps = attns_maps.mean(2)
        # block_num, B, all_num, all_num
        residual_att = torch.eye(attns_maps.size(2)).type_as(attns_maps)
        aug_att_mat = attns_maps + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(-1).unsqueeze(-1)

        joint_attentions = torch.zeros(aug_att_mat.size()).type_as(aug_att_mat)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        attn_proj_map = joint_attentions[-1]
        return joint_attentions

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, cls_score, mask_ins_pred, labels, label_weights):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=None)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_

                losses['acc'] = accuracy(cls_score, labels)

        return losses

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      img_metas, rescale, threshold):

        mask_atten_map = mask_pred.permute(1, 0, 2, 3)
        batch_len = [len(gt) for gt in det_labels]
        mask_atten_map = mask_atten_map.split(batch_len)
        cls_segms_list, im_mask_list = [], []
        for i in range(len(img_metas)):
            scale_factor = img_metas[i]['scale_factor']
            ori_shape = img_metas[i]['ori_shape']
            atten_map = mask_atten_map[i]
            device = mask_pred.device
            cls_segms = [[] for _ in range(self.num_classes)
                         ]  # BG is not included in num_classes
            bboxes = det_bboxes[i][:, :4]
            labels = det_labels[i]
            if not isinstance(scale_factor, torch.Tensor):
                if isinstance(scale_factor, float):
                    scale_factor = np.array([scale_factor] * 4)
                    warn('Scale_factor should be a Tensor or ndarray '
                         'with shape (4,), float would be deprecated. ')
                assert isinstance(scale_factor, np.ndarray)
                scale_factor = torch.Tensor(scale_factor)

            if rescale:
                img_h, img_w = ori_shape[:2]
                bboxes = bboxes / scale_factor
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

            ## block (N,block,W,H) mean to (N,W,H)
            atten_map = atten_map.mean(dim=1)[:, None]
            N = len(atten_map)
            # The actual implementation split the input into chunks,
            # and paste them chunk by chunk.
            if device.type == 'cpu':
                # CPU is most efficient when they are pasted one by one with
                # skip_empty=True, so that it performs minimal number of
                # operations.
                num_chunks = N
            else:
                # GPU benefits from parallelism for larger chunks,
                # but may have memory issue
                # the types of img_w and img_h are np.int32,
                # when the image resolution is large,
                # the calculation of num_chunks will overflow.
                # so we neet to change the types of img_w and img_h to int.
                # See https://github.com/open-mmlab/mmdetection/pull/5191
                num_chunks = int(
                    np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                            GPU_MEM_LIMIT))
                assert (num_chunks <=
                        N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
            chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

            # threshold = 0.01
            im_mask = torch.zeros(
                N,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if threshold >= 0 else torch.uint8)

            for inds in chunks:
                masks_chunk, spatial_inds = _do_paste_mask(
                    atten_map[inds],
                    bboxes[inds],
                    img_h,
                    img_w,
                    skip_empty=device.type == 'cpu')

                if threshold >= 0:
                    masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
                else:
                    # for visualization and debugging
                    masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

                im_mask[(inds,) + spatial_inds] = masks_chunk

            for i in range(N):
                cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
            cls_segms_list.append(cls_segms)
            im_mask_list.append(im_mask)
        return cls_segms_list, im_mask_list

    def onnx_export(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, **kwargs):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor): shape (n, #class, h, w).
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)

        Returns:
            Tensor: a mask of shape (N, img_h, img_w).
        """

        mask_pred = mask_pred.sigmoid()
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX
        img_h, img_w = ori_shape[:2]
        threshold = rcnn_test_cfg.mask_thr_binary
        if not self.class_agnostic:
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, labels][:, None]
        masks, _ = _do_paste_mask(
            mask_pred, bboxes, img_h, img_w, skip_empty=False)
        if threshold >= 0:
            masks = (masks >= threshold).to(dtype=torch.bool)
        return masks


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
