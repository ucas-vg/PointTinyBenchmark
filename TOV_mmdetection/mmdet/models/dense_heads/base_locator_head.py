from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import AnchorFreeHead
from mmcv.runner import BaseModule
from torch import nn
import torch
from mmdet.models.builder import build_loss
from mmdet.core import multi_apply

import skimage.transform  # conda install scikit-image
import numpy as np
from mmdet.utils.locator import utils


class outconv(BaseModule):
    def __init__(self, in_ch, out_ch, init_cfg=None):
        super(outconv, self).__init__(init_cfg)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


@HEADS.register_module()
class BasicLocatorHead(AnchorFreeHead):
    """
        score_map + middle_feat -> regression
        score_feat + middle_feat -> regression
    """
    def __init__(self,
                 num_classes,
                 height,
                 width,
                 counter_fc_channel=128,
                 ultrasmall=False,
                 share_counter=True,
                 loss_count=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_loc=dict(type='WeightedHausdorffDistance', p=-9),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01),
                 *args, **kwargs):
        self.height = height
        self.width = width
        self.ultrasmall = ultrasmall
        self.counter_fc_channel = counter_fc_channel
        assert counter_fc_channel % 2 == 0
        self.share_counter = share_counter
        super().__init__(num_classes, -1, init_cfg=init_cfg, *args, **kwargs)

        self.loss_count = build_loss(loss_count)
        self.loss_loc = build_loss(loss_loc)

    def _init_layers(self):
        self.conv_cls = outconv(64, self.num_classes)
        self.sigmoid = nn.Sigmoid()

        # counter branch
        steps = 3 if self.ultrasmall else 8
        height_mid_features = self.height // (2 ** steps)
        width_mid_features = self.width // (2 ** steps)
        self.count_branch1 = nn.Sequential(
            nn.Linear(height_mid_features * width_mid_features * 512, self.counter_fc_channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        self.count_branch2 = nn.Sequential(
            nn.Linear(self.height * self.width, self.counter_fc_channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        if self.share_counter:
            self.counter = nn.Sequential(
                nn.Linear(self.counter_fc_channel, self.num_classes), nn.ReLU())
        else:
            self.counter = nn.ModuleList([
                nn.Sequential(nn.Linear(self.counter_fc_channel, self.num_classes), nn.ReLU())
                for _ in range(self.num_classes)
            ])

    def forward(self, feats):
        middle_layer, x = feats
        batch_size = x.shape[0]
        x = self.conv_cls(x)
        x = self.sigmoid(x)

        x_flat = x.view(batch_size, self.num_classes, -1)  # B, C, H*W
        lateral_flat = self.count_branch1(middle_layer.view(batch_size, -1))
        regressions = []
        for i in range(self.num_classes):
            x_flat_cls = self.count_branch2(x_flat[:, i])
            regression_features = torch.cat((x_flat_cls, lateral_flat), dim=1)
            if self.share_counter:
                regression = self.counter(regression_features)
            else:
                regression = self.counter[i](regression_features)
            regressions.append(regression)
        regressions = torch.stack(regressions, dim=1)

        return x, regressions

    def bbox_gt_to_point_gt(self, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        gt_points = [(gt_bboxes[i][:, :2] + gt_bboxes[i][:, 2:]) / 2 for i in range(len(gt_bboxes))]
        if gt_bboxes_ignore is not None:
            gt_points_ignore = [(gt_bboxes_ignore[i][:, :2] + gt_bboxes_ignore[i][:, 2:]) / 2 for i in range(len(gt_bboxes_ignore))]
        else:
            gt_points_ignore = None
        return gt_points, gt_labels, img_metas, gt_points_ignore

    def _group_by_class(self, gt_labels, data):
        """
        Args:
            gt_labels: (B, (num_gt, ))
            data: (B, (num_gt, ...)), such as gt_points: (B, (num_gt, 2)
        Returns:
            gt_points_group: (num_classes, B, (num_gt_the_class, 2))
        """
        data_group = [
            [
                [] for _ in range(len(data))
            ] for _ in range(self.num_classes)
        ]

        for img_id, gt_label_per_img in enumerate(gt_labels):
            for gt_id, gt_label in enumerate(gt_label_per_img):
                data_group[gt_label][img_id].append(data[img_id][gt_id])
        return data_group

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass

    def loss(self, est_maps, est_counts, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """
        Args:
            x: est map,
            n_pts:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
        """
        gt_points, gt_labels, img_metas, gt_points_ignore = self.bbox_gt_to_point_gt(
            gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

        gt_points_group, gt_points_ignore_group = [self._group_by_class(gt_labels, data)
                                                   for data in [gt_points, gt_points_ignore]]
        loss_loc1, loss_loc2, loss_count = [], [], []
        for i in range(self.num_classes):
            gt_points_cls, est_maps_cls, est_counts_cls = gt_points_group[i], est_maps[:, i], est_counts[:, i]

            loss1, loss2 = self.loss_loc(est_maps_cls, gt_points_cls)
            gt_count = est_counts_cls.new_tensor([len(pts_cls_img) for pts_cls_img in gt_points_cls])
            loss3 = self.loss_count(est_counts_cls.view(-1), gt_count)

            loss_loc1.append(loss1)
            loss_loc2.append(loss2)
            loss_count.append(loss3)

        return {"loss_loc1": loss_loc1, "loss_loc2": loss_loc2, 'loss_count': loss_count}

    def get_bboxes(self, est_maps, est_counts, img_metas, cfg=None, rescale=False):
        device_cpu = torch.device('cpu')
        # The estimated map must be thresholed to obtain estimated points
        # BMM thresholding
        points = []
        for i in range(len(est_maps)):
            est_map_numpy = est_maps[i, :, :].to(device_cpu).numpy()
            est_count_int = int(round(est_counts[i].item()))

            est_map_numpy = skimage.transform.resize(est_map_numpy, output_shape=est_map_numpy.shape, mode='constant')
            mask, _ = utils.threshold(est_map_numpy, tau=-1)
            # Obtain centroids of the mask
            centroids = utils.cluster(mask, est_count_int, max_mask_pts=np.infty)

            if rescale:
                scale_factor = img_metas[i]['scale_factor']
                centroids = centroids.as_type(np.float32) / scale_factor
            points.append(torch.tensor(centroids).float())

        h, w = cfg.get('pseudo_bbox_wh', (10, 10))
        torch.tensor([h, w])
        return points, det_labels
