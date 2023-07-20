import torch

from mmdet.core.anchor.builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class MyPointGenerator:

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        """
        Args:
            featmap_size:
            stride:
            device:

        Returns:
            all_points: shape=(N, 3), N=H*W, 3 means (x, y, stride)
        """
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid


import torch

from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS


@BBOX_ASSIGNERS.register_module()
class MyPointAssigner(BaseAssigner):
    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """
            points: (n, 3) (x, y, stride)
            gt_bboxes: (k, 4)
            gt_labels: (k,)
            1. assign gts to it's FPN level by it's size
            2. assign every points to the background_label (-1)
            3. A point is assigned to some gt bbox if
                (i) the point is within the k closest points to the gt bbox
                (ii) the distance between this point and the gt is smaller than
                    other gt bboxes
        """
        num_points = len(points)
        num_gts = len(gt_bboxes)
        if num_points == 0 or num_gts == 0:
            assigned_gt_idxs = points.new_full((num_points, ), 0, dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ), -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_idxs, None, labels=assigned_labels
            )

        # get FPN level of points and gt
        points_xy, points_stride = points[:, :2], points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)   # >>>>>>>>>>>>clamp
        # same as FCOS's regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
        scale = self.scale
        gt_bboxes_lvl = (torch.log2(gt_bboxes_wh[:, 0] / scale) + torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        assigned_gt_idxs = points.new_zeros((num_points, ), dtype=torch.long)
        assigned_gt_dist = points.new_zeros((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points_xy[lvl_idx]
            gt_point = gt_bboxes_xy[[idx]]
            gt_wh = gt_bboxes_wh[[idx]]

            points_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            min_dist, min_dist_index = torch.topk(points_dist, self.pos_num, largest=False)

            min_dist_points_index = points_index[min_dist_index]
            find_less_dist = min_dist < assigned_gt_dist[min_dist_points_index]
            find_less_dist_point_idx = min_dist_points_index[find_less_dist]
            assigned_gt_dist[find_less_dist_point_idx] = min_dist[find_less_dist]
            assigned_gt_idxs[find_less_dist_point_idx] = idx + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_idxs.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(assigned_gt_idxs > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_idxs[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_idxs, None, labels=assigned_labels)
