import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from ..match_costs import build_match_cost
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class CostAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self,
                 cls_costs=[dict(type='ClassificationCost', weight=1.)],
                 reg_costs=[dict(type='BBoxL1Cost', weight=1.0, norm_with_img_size=True),
                            dict(type='IoUCost', iou_mode='giou', weight=1.0)],
                 mode='topk', k=1
                 ):
        cls_costs = cls_costs if isinstance(cls_costs, (tuple, list)) else [cls_costs]
        reg_costs = reg_costs if isinstance(reg_costs, (tuple, list)) else [reg_costs]
        self.cls_costs = [build_match_cost(cls_cost) for cls_cost in cls_costs]
        self.reg_costs = [build_match_cost(reg_cost) for reg_cost in reg_costs]
        self.mode = mode
        self.k = k

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. chosen positive with costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2, ...),. Shape [num_query, k*2].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2, ...). Shape [num_gt, k*2].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        cls_costs = [cls_cost(cls_pred, gt_labels) for cls_cost in self.cls_costs]
        reg_costs = [reg_cost(bbox_pred, gt_bboxes, img_meta) for reg_cost in self.reg_costs]
        cost = sum(cls_costs) + sum(reg_costs)

        # 3. chose positive
        _, top_k_idx = cost.topk(self.k, dim=0, largest=False)
        cost[top_k_idx, torch.arange(cost.shape[-1]).view(1, -1).repeat(self.k, 1)]

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
