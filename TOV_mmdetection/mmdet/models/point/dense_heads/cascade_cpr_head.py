import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.models.builder import build_head
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from copy import deepcopy
import mmcv.runner.base_module as mmcv_base

from mmcv.runner import BaseModule
from mmdet.models.losses import accuracy
from collections import defaultdict
from mmdet.core.point.cpr_utils.utils import CascadeData
from mmdet.core.point.cpr_utils.test_cpr import TestCPRHead


def update_dict(old_d, new_d):
    nd = deepcopy(old_d)
    for k, v in new_d.items():
        if isinstance(v, dict) and k in old_d:
            nd[k] = update_dict(old_d[k], v)
        else:
            nd[k] = v
    return nd


@HEADS.register_module()
class CascadeCPRHead(AnchorFreeHead):

    def __init__(self,
                 num_classes, in_channels,
                 cascade_cfg=dict(
                     gt_src='gt_refine',
                     weight_with_score=False,
                     weight_type='max',
                     conditional_refine=False,
                     increase_r=False,
                     increase_r_step=1,
                     share_feat=False,
                     mean_loss=True,
                     detach_feat_for_refine_point=True,
                 ),
                 num_stages=-1,
                 cpr_cfg_list=[],
                 init_cfg=None,
                 **shared_kwargs):
        """
        build cpr heads:
            1. num_stages
            2. cpr_cfg_list
            3. sub_head0, sub_head1...
        """
        shared_kwargs = deepcopy(shared_kwargs)
        if num_stages > 0:
            assert cpr_cfg_list is None or len(cpr_cfg_list) == 0
            cpr_cfg_list = [dict(type='CPRHead') for _ in range(num_stages)]
        assert isinstance(cpr_cfg_list, (list, tuple))
        if num_stages <= 0 and len(cpr_cfg_list) == 0:
            cpr_cfgs = {}
            for key, value in list(shared_kwargs.items()):
                if key.startswith("sub_head"):
                    cpr_cfgs[int(key[len('sub_head'):])] = shared_kwargs.pop(key)
            assert sorted(list(cpr_cfgs.keys())) == list(range(len(cpr_cfgs))), f"{cpr_cfgs.keys()}"
            cpr_cfg_list.extend([cpr_cfgs[i] for i in range(len(cpr_cfgs))])

        super(AnchorFreeHead, self).__init__(init_cfg)

        self.cascade_cfg = cascade_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        shared_kwargs.update(dict(num_classes=num_classes, in_channels=in_channels))

        from mmdet.utils.logger import get_root_logger
        logger = get_root_logger()

        self.cpr_list = mmcv_base.ModuleList()
        for i, cpr_cfg in enumerate(cpr_cfg_list):
            base_cfg = deepcopy(shared_kwargs)
            cpr_cfg = update_dict(base_cfg, cpr_cfg)
            cpr_cfg = update_dict(cpr_cfg, self.incremental_r_cfg(shared_kwargs, i))

            if i < len(cpr_cfg_list) - 1:
                cpr_cfg['point_refiner']['return_score_type'] = self.cascade_cfg['weight_type']
            if i > 0 and self.cascade_cfg['share_feat']:
                cpr_cfg['stacked_convs'] = 0
                # if have KeyError on 'type', check whether give wrong $i for args in sub_head$i
                if cpr_cfg['type'] in ['CPRHead', 'CPRV2Head']:
                    cpr_cfg['stacked_fcs'] = 0

            logger.info(f"[CPR Head {i}]: {cpr_cfg}")
            cpr = build_head(cpr_cfg)
            self.cpr_list.append(cpr)

    def incremental_r_cfg(self, shared_kwargs, stage_i):
        if not self.cascade_cfg['increase_r']:
            return {}
        t_pos_r = shared_kwargs["train_pts_extractor"]['pos_generator']['radius']
        t_neg_r = shared_kwargs["train_pts_extractor"]['neg_generator']['radius']
        r_pos_r = shared_kwargs["refine_pts_extractor"]['neg_generator']['radius']
        r_neg_r = shared_kwargs["refine_pts_extractor"]['neg_generator']['radius']

        r = stage_i * self.cascade_cfg['increase_r_step']
        return dict(
            train_pts_extractor=dict(pos_generator=dict(radius=t_pos_r+r), neg_generator=dict(radius=t_neg_r+r)),
            refine_pts_extractor=dict(pos_generator=dict(radius=r_pos_r+r), neg_generator=dict(radius=r_neg_r+r))
        )

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None,
                      proposal_cfg=None, **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_true_bboxes=gt_true_bboxes)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False, **kwargs):
        # TestCPRHead.show_feats(feats, img_metas)
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, **kwargs)
        return results_list

    def forward(self, feats):
        if self.cascade_cfg['share_feat']:
            if not hasattr(self.cpr_list[0], "share_forward"):  # old version
                res = [self.cpr_list[0](feats)] * len(self.cpr_list)
            else:
                share_part = self.cpr_list[0].share_forward(feats)
                res = [cpr.unshare_forward(share_part, feats) for cpr in self.cpr_list]
        else:
            res = [cpr(feats) for cpr in self.cpr_list]
        return res,

    def update_with_refine_outs(self, refine_outs, gt_r_bboxes, gt_labels, gt_weights, not_refine, ret_cascade_data):
        """
            'gt_r_bboxes': gt_src, decide gt_r_bbox = cat_or_not(refine_bboxes, gt_r_bbox)
            'gt_weight': weight_with_score or not
            'not_refine': conditional refine
        """
        refine_res, cur_not_refine = refine_outs
        refine_bboxes = [rb[..., :4] for rb, gt_l in refine_res]
        refine_scores = [rb[..., 4] for rb, gt_l in refine_res]
        refine_labels = [gt_l for rb, gt_l in refine_res]

        for rl, gl in zip(refine_labels, gt_labels):
            assert (rl == gl).all()

        if self.cascade_cfg['gt_src'] == 'gt_refine':
            gt_r_bboxes = [gb.reshape(len(labels), -1, *gb.shape[1:]) for gb, labels in zip(gt_r_bboxes, gt_labels)]
            gt_r_bboxes = [torch.cat([grb[:, :1], rb.unsqueeze(dim=1)], dim=1).flatten(0, 1)
                           for rb, grb in zip(refine_bboxes, gt_r_bboxes)]
        elif self.cascade_cfg['gt_src'] == 'refine':
            gt_r_bboxes = refine_bboxes
        elif self.cascade_cfg['gt_src'] == 'gt':
            pass
        else:
            raise ValueError

        if self.cascade_cfg['weight_with_score']:
            gt_weights = [s for s in refine_scores]

        if self.cascade_cfg['conditional_refine']:
            not_refine = cur_not_refine
        return gt_r_bboxes, gt_weights, not_refine, ret_cascade_data

    def loss(self, feats, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, gt_true_bboxes=None):
        def detach(x_list):
            if isinstance(x_list, torch.Tensor):
                return x_list.detach()
            elif isinstance(x_list, (tuple, list)):
                return [detach(x) for x in x_list]
            else:
                raise TypeError(type(x_list))

        all_losses = {}
        gt_r_bboxes = gt_bboxes
        gt_anns_id = [torch.LongTensor([-1] * len(l)) for l in gt_labels]
        feats_detach = detach(feats) if self.cascade_cfg["detach_feat_for_refine_point"] else feats
        gt_weights = None
        not_refine = None
        cascade_data = CascadeData()

        for i, cpr in enumerate(self.cpr_list):
            losses = cpr.loss(*feats[i], gt_r_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore, gt_true_bboxes, gt_weights, cascade_refine_data=cascade_data)

            for key, v in losses.items():
                if self.cascade_cfg['mean_loss'] and key.find("loss") > 0:  # mean all loss
                    v = v / len(self.cpr_list)
                all_losses[key + f"_{i}"] = v

            if i == len(self.cpr_list) - 1:
                break

            refine_res, cur_not_refine, ret_data = cpr.get_bboxes(
                *feats_detach[i], img_metas, rescale=False, gt_bboxes=gt_r_bboxes,
                gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore, gt_true_bboxes=gt_true_bboxes,  # for debug
                gt_anns_id=gt_anns_id, not_refine=not_refine, cascade_out_fmt=True, cascade_refine_data=cascade_data
            )

            gt_r_bboxes, gt_weights, not_refine, cascade_data = self.update_with_refine_outs(
                (refine_res, cur_not_refine), gt_r_bboxes, gt_labels, gt_weights, not_refine, ret_data)

        return all_losses

    def get_bboxes(self, feats, img_metas,
                   cfg=None, rescale=False, with_nms=True,
                   gt_bboxes=None, gt_labels=None, gt_bboxes_ignore=None, gt_true_bboxes=None, gt_anns_id=None):

        gt_r_bboxes = gt_bboxes
        not_refine = None
        cascade_data = CascadeData()

        for i, cpr in enumerate(self.cpr_list):
            sub_rescale = rescale if i == len(self.cpr_list) - 1 else False
            refine_res, cur_not_refine, ret_data = cpr.get_bboxes(
                *feats[i], img_metas, rescale=sub_rescale, gt_bboxes=gt_r_bboxes,
                gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore, gt_true_bboxes=gt_true_bboxes,  # for debug
                gt_anns_id=gt_anns_id, not_refine=not_refine, cascade_out_fmt=True, cascade_refine_data=cascade_data
            )
            if i == len(self.cpr_list) - 1:
                break

            gt_r_bboxes, _, not_refine, cascade_data = self.update_with_refine_outs(
                (refine_res, cur_not_refine), gt_r_bboxes, gt_labels, None, not_refine, ret_data)
        return refine_res

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass


if __name__ == '__main__':
    # [-0.5, w-1+0.5] -> [-1, 1]
    # x -> x' => x' = (2x+1) / w - 1
    grid_map_func = lambda xy, wh: (2 * xy + 1) / wh - 1
    input = torch.arange(16).reshape(1, 1, 4, 4).float()
    # grid = torch.tensor([[[
    #     [-1, -1], [-0.5, -0.5], [0, 0], [0.5, 0.5], [1, 1]
    # ]]]).float()
    grid = torch.tensor([[[
        [0, 0], [1, 1], [2, 2]
    ]]]).float()
    grid = grid_map_func(grid, grid.new_tensor([input.shape[-1], input.shape[-2]]))
    x = F.grid_sample(input, grid, padding_mode='border', align_corners=False)
    print(input)
    print(x)

    # [0, w-1] -> [-1, 1]
    grid_map_func = lambda xy, wh: 2*xy / (wh-1) - 1
    grid = torch.tensor([[[
        [0, 0], [1, 1], [2, 2]
    ]]]).float()
    grid = grid_map_func(grid, grid.new_tensor([input.shape[-1], input.shape[-2]]))
    x = F.grid_sample(input, grid, padding_mode='border', align_corners=True)
    print(input)
    print(x)

    from PIL import Image
    import matplotlib.pyplot as plt

    # k line: X[:, k], Y[:, k]
    plt.plot([[0, 1], [2, 3]], [[8, 9], [10, 4]])
    plt.plot()
    plt.show()

    # print(torch.topk(torch.tensor([[1, 2, 30, 9, 7, 6], [1, 2, 30, 9, 7, 10]]), 3, largest=False, dim=1))
    #
    # print(torch.tensor([[1, 2, 3], [4, 5, 6]]).float().norm(dim=-1))
