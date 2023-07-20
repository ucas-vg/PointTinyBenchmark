"""
1. offset_to_pts 网络forward输出的pts_pred是9个点相对于预测点的offset，转换为9个点在图片中的位置：pts_pred ->
    centers: (B, num_levels, (H*W, 3))  # 3 is (x, y, stride)
        point_generator.grid_points
        point_generator.valid_flags

2. points_to_bbox 将预测的pts转换为 box (x1, y1, x2, y2): pred_pts->pred_bbox

3. assigner.assgin 根据gt_bboxes给每个点分配label
    init stage   : center   , gt_bboxes
    refine stage : bbox_init.detach(), gt_bboxes

loss --> loss_single --> get_targets --> get_targets_single --> assign --> sample

pts_pred_init + centers -> pred_init_pts
gts & centers -> labels,
pts_pred_refine + centers -> pred_refine_pts
"""

from MyRepPointsHead3 import MyRepPointsHead3
from Data import gen_pseudo_inputs

from mmdet.models.builder import build_loss
from mmdet.core import PointGenerator, multi_apply
import numpy as np
import torch
from mmdet.core import build_assigner, build_sampler


def merge_dims(data, dims):
    assert tuple(range(dims[0], dims[-1] + 1)) == tuple(dims), "merge dims must be continue interger"
    shape = data.shape
    new_shape = list()
    return data.reshpae(*shape[:dims[0]], -1, *shape[dims[-1] + 1:])


class MyRepPointsHead4(MyRepPointsHead3):
    def __init__(self, num_classes, in_channels,
                 point_strides=[8, 16, 32, 64, 128],
                 *args, **kwargs):

        super().__init__(num_classes, in_channels, *args, **kwargs)

        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

    def get_points(self, featmap_sizes, img_metas, device):
        """
            point_generator.grid_points
            point_generator.valid_flags
            center: (B, num_levels, (H*W, 3))  # 3 is (x, y, stride)
        """
        num_levels = len(featmap_sizes)
        num_imgs = len(img_metas)

        multi_level_points = [self.point_generators[i].grid_points(featmap_sizes[i], self.point_strides[i], device) \
                              for i in range(num_levels)]
        points_list = [[points.clone() for points in multi_level_points] for i in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                s = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]  # ??? pad_shape
                valid_feat_h = min(int(np.ceil(h / s)), feat_h)
                valid_feat_w = min(int(np.ceil(w / s)), feat_w)
                flags = self.point_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """
            center_list: (B, num_levels, (H*W, 3))  # 3 is (x, y, stride)
            pred_list: (num_levels, (B, C, H, W))   # C is (y0_off, x0_off, y1_off, x0_off, ...) follow offsets of DConv
            pts_list: (num_levels, (B, H*W, num_points*2))
        """
        pts_list = []
        for i in range(len(self.point_strides)):
            pts_lvl = []
            for img_id in range(len(center_list)):
                center = center_list[img_id][i][:, :2].repeat(1, self.num_points)
                pts_offset = pred_list[i][img_id].permute(1, 2, 0).view(-1, 2 * self.num_points)
                y_offset, x_offset = pts_offset[:, 0::2], pts_offset[:, 1::2]
                xy_offset = torch.stack([x_offset, y_offset], dim=-1).view(-1, 2 * self.num_points)
                pts = center + xy_offset * self.point_strides[i]
                pts_lvl.append(pts)
            pts_list.append(torch.stack(pts_lvl, 0))
        return pts_list

    def points2bbox(self, pts, y_first=True):
        """
            pts: (X, num_points*2, ...)
        """
        pts = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts[:, :, 0, ...] if y_first else pts[:, :, 1, ...]
        pts_x = pts[:, :, 1, ...] if y_first else pts[:, :, 0, ...]

        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        return bbox

    def loss(self, cls_scores, pts_preds_init, pts_preds_refine,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        num_levels, B, device = len(cls_scores), cls_scores[0].shape[0], cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)
        pred_xy_init = self.offset_to_pts(center_list, pts_preds_init)

        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas, device)
        pred_xy_refine = self.offset_to_pts(center_list, pts_preds_refine)

        bbox_shifts = [self.points2bbox(pts_preds_init[i].detach()) * stride for i, stride in
                       enumerate(self.point_strides)]
        bbox_init_list = []
        for img_id in range(B):
            center = center_list[img_id]
            multi_lvl_bbox = []
            for i in range(num_levels):
                bbox = center[i][:, :2].repeat(1, 2) + bbox_shifts[i][img_id].permute(1, 2, 0).view(-1, 4)
                multi_lvl_bbox.append(bbox)
            bbox_init_list.append(multi_lvl_bbox)
        return center_list, valid_flag_list, bbox_init_list, pred_xy_init, pred_xy_refine


if __name__ == '__main__':
    inputs = gen_pseudo_inputs(2, 256, 2 * 32, 3 * 32, 5, device=0)

    my_model = MyRepPointsHead4(
        num_classes=80,
        in_channels=256).to(0)

    res = my_model(inputs[0])
    res = my_model.loss(*res, *inputs[1:])

