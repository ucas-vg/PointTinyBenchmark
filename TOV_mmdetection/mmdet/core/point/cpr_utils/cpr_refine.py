import matplotlib.hatch
import matplotlib.pyplot as plt

from mmdet.core.point.cpr_utils.test_cpr import TestCPRHead
import torch
from mmdet.core.point.cpr_utils.utils import group_by_label, PtAndFeat, CascadeData, Statistic
from mmdet.core import multi_apply


class PointRefiner(object):
    def __init__(self, strides, gt_alpha=0.5, merge_th=0.05, refine_th=0.05,
                 classify_filter=False, refine_pts_extractor=None, return_score_type='mean',
                 nearest_filter=True, out_geo_type='bbox',
                 debug_cfg=dict(open=False)):
        self.gt_alpha = gt_alpha
        self.merge_th = merge_th
        self.refine_th = refine_th
        self.strides = strides
        self.use_classify_filter = classify_filter
        self.use_nearest_filter = nearest_filter

        self.refine_pts_extractor = refine_pts_extractor
        # self.pos_generator = build_from_type(pos_generator)
        # self.other_generators = [build_from_type(g) for g in other_generators]
        self.return_score_type = return_score_type
        self.out_geo_type = out_geo_type
        self.debug_cfg = debug_cfg
        print(debug_cfg)
        TestCPRHead.DO_TEST = debug_cfg["open"]
        TestCPRHead.DBI['COUNT'] = debug_cfg.get('COUNT', -1)
        TestCPRHead.DBI['epoch'] = debug_cfg.get('epoch', -1)
        TestCPRHead.DBI['show'] = debug_cfg.get('show', True)
        self.statistic = Statistic()
        self.stage = debug_cfg.get('stage', -1)

    def grid_merge_per_class(self, grid_cls_prob, dist, gt_prob, gt_r_points, num_refine):
        valid = (grid_cls_prob > self.merge_th) & (grid_cls_prob > gt_prob * self.gt_alpha)
        _, closest_gt_idx = dist[valid].min(dim=1)
        chosen_pts = []
        for idx in closest_gt_idx:
            gt_idx = idx % num_refine

    # def grid_merge(self, grid_cls_prob, grid_pts, gt_r_pts, gt_labels, num_refine):
    #     """
    #     1. assign grid point to each object
    #     Returns:
    #     """
    #     grid_pts, grid_cls_prob = grid_pts.flatten(0, -2), grid_cls_prob.flatten(0, -2)
    #     dist = torch.cdist(grid_pts[..., :2], gt_r_pts[..., :2], p=2)
    #
    #     cls2gt_idx = defaultdict(list)
    #     for i, l in enumerate(gt_labels):
    #         cls2gt_idx[l].append(i)
    #     for l, gt_idx in cls2gt_idx.items():
    #         gt_r_pts_l = gt_r_pts[gt_idx]
    #         dist_l = dist[:, gt_idx]
    #         grid_cls_prob_l = grid_cls_prob[:, l]

    def nearest_filter(self, bag_pts, gt_r_pts, gt_labels, class_wise=True):
        """
        Args:
            bag_pts: shape=(num_gts, num_refine, num_chosen, 3)
            gt_r_pts:  (num_gts, num_refine, 2)
            gt_labels: (num_gts,)
            class_wise:
        Returns:
        """

        def filter(bag_pts, gt_r_pts):
            num_gts, num_refine, num_chosen, _ = bag_pts.shape
            dist = torch.cdist(bag_pts.flatten(0, -2)[..., :2], gt_r_pts.flatten(0, -2)[..., :2], p=2)
            _, closest_gt_idx = dist.min(dim=1)
            closest_gt_idx = closest_gt_idx.reshape(num_gts * num_refine, num_chosen)
            cur_gt_idx = torch.arange(len(closest_gt_idx)).reshape(-1, 1).to(closest_gt_idx.device)
            close_valid = (closest_gt_idx == cur_gt_idx).reshape(num_gts, num_refine * num_chosen)
            return close_valid

        if class_wise:
            gt_idx = torch.arange(len(bag_pts))
            label2gt_r_pts = group_by_label(gt_r_pts, gt_labels)
            label2bag_pts = group_by_label(bag_pts, gt_labels)
            label2gt_idx = group_by_label(gt_idx, gt_labels)

            num_gts, num_refine, num_chosen, _ = bag_pts.shape
            valid = gt_labels.new_ones((num_gts, num_refine * num_chosen), dtype=torch.bool)
            for l in label2gt_r_pts:
                gt_r_pts, bag_pts, gt_idx = label2gt_r_pts[l], label2bag_pts[l], label2gt_idx[l]
                if len(gt_r_pts) > 1:
                    valid[gt_idx] = filter(bag_pts, gt_r_pts)
            return valid
        else:
            return filter(bag_pts, gt_r_pts)

    def classify_filter(self, bag_cls_prob, gt_labels):
        """
        Args:
            bag_cls_prob:  (num_gts, ..., num_class)
            gt_labels: (num_gts, )
        Returns:
        """
        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        _, classify_res = bag_cls_prob.max(dim=-1)
        shape = [len(classify_res)] + [1] * (len(classify_res.shape) - 1)
        valid = classify_res == gt_labels.reshape(*shape)
        return valid.reshape(num_gts, num_refine * num_chosen)

    def graph_filter(self, strides, bag_valid):
        """
        Args:
            strides: (..., 1)
            bag_valid: (num_gts, num_refine, num_chosen)
        Returns:
        """
        stride = strides.reshape(-1, 1)[0].tolist()[0]
        assert (strides == stride).all(), ""
        gt_valid = bag_valid[..., -1].clone()
        bag_valid[..., -1] = True
        self.refine_pts_extractor.pos_generator.filter_depend(stride, 2.0, bag_valid)
        bag_valid[..., -1] = gt_valid
        return bag_valid.flatten(0, 1)

    def inside_img(self, bag_pts, img_shape):
        num_gts, num_refine, num_chosen, _ = bag_pts.shape
        bag_pts = bag_pts.reshape(num_gts, num_refine * num_chosen, -1)
        h, w, _ = img_shape
        x, y = bag_pts[..., 0], bag_pts[..., 1]
        return (x < w) & (x >= 0) & (y < h) & (y >= 0)

    def statistic_scores(self, bag_cls_prob, gt_cls_prob, num_lvl):
        self.statistic.print_mean("score of points in bag", bag_cls_prob.flatten(), log_interval=500)
        self.statistic.print_mean("score of points in gt", gt_cls_prob.flatten(), log_interval=500)

        for th in [0.1, 0.2, 0.3]:
            rate = (bag_cls_prob > th).sum().float() / (bag_cls_prob > 0).sum()
            self.statistic.print_mean(f"score > {th} point rate in bag", rate.flatten(), log_interval=500)
        # num_gts, num_chosen = bag_cls_prob.shape  # num_refine must == 1
        # bag_cls_prob = bag_cls_prob.reshape(num_gts, num_lvl, num_chosen//num_lvl)
        # max_score = bag_cls_prob.max(dim=2)[0]
        # self.statistic.print_mean(f"max score of each level in bag", max_score, log_interval=500)

    def statistic_after_refine(self, bag_pts_weight, bag_cls_prob, chosen_pts, bag_pts, not_refine,
                               pts_lvl, num_lvl):
        if self.debug_cfg.get('refine_statistic', False):
            self.statistic.print_mean("score of chosen points", bag_cls_prob[bag_cls_prob > 0], log_interval=500)
            num_chosen_pts = [len(pts) for pts in chosen_pts if len(pts) > 1]
            if len(num_chosen_pts) == 0:
                num_chosen_pts = [1]
            self.statistic.print_mean("number of chosen points of refined", torch.tensor(num_chosen_pts),
                                      log_interval=500)
            self.statistic.print_mean("refine rate", 1 - not_refine.float(), log_interval=500)
        if self.debug_cfg.get("lvl_statistic", False):
            pass
            # lvl = [pts_lvl[chosen] for gt_i, chosen in enumerate(bag_pts_weight > 0)]
            # lvl_count = torch.tensor([[0.0] * num_lvl for _ in lvl]).float()
            # for gt_i, gt_lvl in enumerate(lvl):
            #     for l in gt_lvl:
            #         lvl_count[gt_i, l] += 1.0
            # sum_count = lvl_count.sum(dim=-1, keepdim=True)+1e-7
            # self.statistic.print_mean("number of chosen points in each lvl", lvl_count, log_interval=500)
            # self.statistic.print_mean("rate of chosen points in each lvl",
            #                           lvl_count/sum_count, log_interval=500)
            # self.statistic.print_log_mean_exp("rate2 of chosen points in each lvl",
            #                                   lvl_count/sum_count, log_interval=500)

    def get_map(self, bag_data, grid_data, gt_labels, gt_true_bboxes, img_meta):
        if not self.debug_cfg.get("variance", False):
            return
        assert len(grid_data.cls_prob) == 1, "only support single level."
        pad_img = TestCPRHead.load_pad_img(img_meta)
        label2bboxes = group_by_label(gt_true_bboxes, gt_labels)
        for l in label2bboxes:
            data = grid_data.cls_prob[0][:, :, l].detach().cpu().numpy()
            mask = (data - data.min()) / (data.max() - data.min())
            mask_img = TestCPRHead.mask_img(mask, pad_img)
            plt.title(f"label: {l}")
            plt.imshow(mask_img)
            plt.show()

        pts = grid_data.pts[0].reshape(-1, 3)[:, :2]
        for l, bboxes in label2bboxes.items():
            score_map = grid_data.cls_prob[0][:, :, l].reshape(-1)
            for bbox in bboxes:
                idx = (pts[:, 0] < bbox[2]) & (pts[:, 0] > bbox[0]) & (pts[:, 1] < bbox[3]) & (pts[:, 1] > bbox[1])
                scores = score_map[idx]
                scores /= scores.sum()
                std_scores = scores.std()
                # std_scores01 = scores[scores > 0.1].std()

                std_scores = std_scores.detach().cpu().numpy()
                # std_scores01 = std_scores01.detach().cpu().numpy()
                print("std scores", self.statistic.mean_scalar("mean std scores", std_scores),
                      self.statistic.min_scalar("min std scores", std_scores),
                      self.statistic.max_scalar("max std scores", std_scores))
                # print("std scores01", self.statistic.mean_scalar("mean std scores01", std_scores01),
                #       self.statistic.min_scalar("min std scores01", std_scores01),
                #       self.statistic.max_scalar("max std scores01", std_scores01))

    def handle_fpn(self, bag_data, gt_labels, gt_true_bboxes):
        """
            bag_cls_prob: [lvl, (num_gts, num_refine, num_chosen, num_class)]
            bag_ins_prob: [lvl, (num_gts, num_refine, num_chosen, num_class)]
        """

        def static(max_lvl):
            # true_bboxes = torch.cat(gt_true_bboxes, dim=0)
            wh = gt_true_bboxes[:, 2:] - gt_true_bboxes[:, :2]
            size = ((wh[:, 0] * wh[:, 1]) ** 0.5).detach().cpu().numpy()
            # print lvl size
            assert num_refine == 1
            max_lvl = max_lvl.detach().cpu().numpy()
            size_lvl = [[] for _ in range(num_lvl)]
            lvl_count = [0] * num_lvl
            for s, l in zip(size, max_lvl):
                size_lvl[l].append(s)
                lvl_count[l] += 1
            self.statistic.print_mean("lvl count", torch.tensor([lvl_count]).float(), log_interval=500)
            self.statistic.print_mean_list("lvl size", size_lvl, log_interval=500)
            self.statistic.print_log_mean_exp_list("lvl log_mean_exp size", size_lvl, log_interval=500)

        bag_ins_prob = bag_data.ins_prob
        if bag_ins_prob is None:
            bag_ins_prob = [ins_out.softmax(dim=-2) for ins_out in bag_data.ins_outs]

        # fpn_select = dict(type='all')
        fpn_select = dict(type='topk', k=3, prob='ins*cls')  # normal ins*cls is same as cls
        # fpn_select = dict(type='topk', k=3, prob='cls')
        if fpn_select['type'] == 'all':
            bag_cls_prob, bag_ins_prob = torch.cat(bag_data.cls_prob, dim=2), torch.cat(bag_ins_prob, dim=2)
            bag_pts, bag_valid = torch.cat(bag_data.pts, dim=2), torch.cat(bag_data.valid, dim=2)
            # grid_cls_prob = torch.cat(grid_data.cls_prob, dim=0)
            # grid_pts, grid_valid = torch.cat(grid_data.pts, dim=0), torch.cat(grid_data.valid, dim=0)
            return bag_cls_prob, bag_ins_prob, bag_valid, bag_pts
        elif fpn_select['type'] == 'topk':
            topk = fpn_select['k']
            bag_cls_prob = torch.stack(bag_data.cls_prob, dim=2)
            bag_ins_prob = torch.stack(bag_ins_prob, dim=2)
            bag_valid = torch.stack(bag_data.valid, dim=2)
            bag_pts = torch.stack(bag_data.pts, dim=2)

            num_gts, num_refine, num_lvl, num_chosen, num_class = bag_cls_prob.shape
            n = num_gts * num_refine
            bag_cls_prob, bag_ins_prob = bag_cls_prob.flatten(0, 1), bag_ins_prob.flatten(0, 1)
            bag_valid, bag_pts = bag_valid.flatten(0, 1), bag_pts.flatten(0, 1)

            if fpn_select['prob'] == 'ins*cls':
                prob = bag_ins_prob * bag_cls_prob * bag_valid.float()
            elif fpn_select['prob'] == 'cls':
                prob = bag_cls_prob * bag_valid.float()
            else:
                raise ValueError
            prob = prob[torch.arange(n), ..., gt_labels]  # (num_gts*num_refine, num_lvl, num_chosen)
            max_v, max_lvl = torch.topk(prob, topk, dim=-1)[0].mean(dim=-1).max(dim=-1)  # (num_gts*num_refine,)

            bag_cls_prob = bag_cls_prob[torch.arange(n), max_lvl].reshape(num_gts, num_refine, *bag_cls_prob.shape[2:])
            bag_ins_prob = bag_ins_prob[torch.arange(n), max_lvl].reshape(num_gts, num_refine, *bag_ins_prob.shape[2:])
            bag_valid = bag_valid[torch.arange(n), max_lvl].reshape(num_gts, num_refine, *bag_valid.shape[2:])
            bag_pts = bag_pts[torch.arange(n), max_lvl].reshape(num_gts, num_refine, *bag_pts.shape[2:])

            if self.debug_cfg.get("lvl_statistic", False):
                static(max_lvl)
            return bag_cls_prob, bag_ins_prob, bag_valid, bag_pts
        else:
            raise ValueError(fpn_select)

    def refine_single(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_r_points, gt_labels,
                      img_meta, gt_true_bboxes, not_refine=None, cascade_refine_data=None):
        """
        refine point in single image
        Args:
            bag_data.cls_prob: [lvl, (num_gts, num_refine, num_chosen, num_class)]
            bag_data.valid: [lvl, (num_gts, num_refine, num_chosen, 1)]
            grid_data.cls_prob: [lvl, (h, w, num_class)]
            grid_data.valid: [lvl, (h, w, num_class)]
            gt_labels: (num_gts,)
            gt_r_points: (num_gts, num_refine, 2)
        Returns:
        """
        # pts_lvl = bag_ins_prob[0].new_tensor(range(num_lvl)).unsqueeze(dim=-1)
        # .repeat(1, num_chosen_lvl).long().flatten()
        self.get_map(bag_data, grid_data, gt_labels, gt_true_bboxes, img_meta)

        # TestCPRHead.get(self).show_scores(bag_data, gt_labels, img_meta)
        num_lvl = len(bag_data.cls_prob)
        bag_cls_prob, bag_ins_prob, bag_valid, bag_pts = self.handle_fpn(bag_data, gt_labels, gt_true_bboxes)

        if self.debug_cfg.get('ins_statistic', False):
            num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
            t_idx = torch.arange(num_gts).to(bag_ins_prob.device)
            bag_ins_score = bag_ins_prob[t_idx, :, :, gt_labels].flatten(0, 1)
            bag_cls_score = bag_cls_prob[t_idx, :, :, gt_labels].flatten(0, 1)
            idx = bag_cls_score.argsort(dim=1)
            idx0 = torch.arange(idx.shape[0]).unsqueeze(dim=1).repeat(1, idx.shape[1])
            self.statistic.print_log_mean_exp("ins_score", bag_ins_score[idx0, idx])

        # 2. split gt prob out, last one in bag
        gt_cls_prob = bag_cls_prob[..., -1:, :]
        gt_r_pts = bag_pts[..., -1:, :]
        center_pts = bag_pts[..., -2, :2]
        assert (gt_r_pts[:, :, -1, :2] == gt_r_points[:,
                                          :1]).all(), f"last one of chosen dim should be annotation point."

        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        gt_idx = torch.arange(len(gt_labels))
        merge_valid = bag_valid.reshape(num_gts, num_refine * num_chosen).bool()
        # 3. assign point
        if self.use_nearest_filter:
            merge_valid &= self.nearest_filter(bag_pts, gt_r_pts, gt_labels)
        if self.use_classify_filter:
            merge_valid &= self.classify_filter(bag_cls_prob, gt_labels)

        # 4. prob > th & prob > gt_prob * a
        bag_cls_prob = bag_cls_prob[gt_idx, ..., gt_labels].reshape(num_gts, num_refine * num_chosen)
        gt_cls_prob = gt_cls_prob[gt_idx, 0, ..., gt_labels].reshape(num_gts, 1)
        if self.debug_cfg.get('refine_statistic', False):
            self.statistic_scores(bag_cls_prob * merge_valid.float(), gt_cls_prob, num_lvl)
        # gt_cls_prob = gt_cls_prob.repeat(1, 1, num_chosen).flatten(1)
        merge_valid &= (bag_cls_prob > self.merge_th) & (bag_cls_prob > gt_cls_prob * self.gt_alpha)
        # graph filter
        # merge_valid &= self.graph_filter(bag_pts[..., -1:], merge_valid.reshape(num_gts, num_refine, num_chosen))
        # inside image filter
        merge_valid &= self.inside_img(bag_pts, img_meta['img_shape'])

        # 5. merge
        bag_pts = bag_pts.reshape(num_gts, num_refine * num_chosen, 3)
        bag_cls_prob = bag_cls_prob * merge_valid.float()
        bag_pts_weight = bag_cls_prob / (bag_cls_prob.sum(dim=1, keepdim=True) + 1e-8)
        refine_pts = (bag_pts[..., :2] * bag_pts_weight.unsqueeze(dim=-1)).sum(dim=1)  # (num_gts, 2)

        refine_scores = bag_cls_prob.sum(dim=-1) / ((bag_cls_prob > 0).float().sum(dim=-1) + 1e-8)  # (num_gts, )
        cur_not_refine = (refine_scores < self.refine_th)  # to prevent number of chosen points is 0
        not_refine = cur_not_refine if not_refine is None else not_refine | cur_not_refine
        refine_pts[not_refine] = gt_r_points[:, 0][not_refine]

        if self.return_score_type == 'max':
            refine_scores = bag_cls_prob.max(dim=-1)[0]
            refine_scores[refine_scores == 0] = self.refine_th / 2
        elif self.return_score_type == 'mean':
            pass
        else:
            raise ValueError

        # shape: [len_gt, (K, 3)]
        chosen_pts = [bag_pts[gt_i][chosen] for gt_i, chosen in enumerate(bag_pts_weight > 0)]
        self.statistic_after_refine(bag_pts_weight, bag_cls_prob, chosen_pts, bag_pts, not_refine, None, num_lvl)
        return refine_pts, refine_scores, not_refine, center_pts, chosen_pts, self.get_geo_output(chosen_pts, refine_pts)

    def get_geo_output(self, chosen_pts, refine_pts):
        geos = []
        for gt_i, (c_pts, r_pt) in enumerate(zip(chosen_pts, refine_pts)):
            if self.out_geo_type == 'bbox':
                if len(c_pts) > 0:
                    x, y = c_pts[:, 0], c_pts[:, 1]
                    bbox = [x.min(), y.min(), x.max(), y.max()]
                else:
                    x, y = refine_pts[gt_i, :2]
                    bbox = [x, y, x, y]
                geos.append(bbox)
            else:
                raise ValueError()
        return refine_pts.new_tensor(geos)

    def __call__(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_r_points, gt_labels, img_metas,
                 gt_true_bboxes, not_refine=None, cascade_refine_data=None):
        """
        1.
        Args:
            bag_data.cls_prob: [num_lvl, (num_gts_all_img, num_refine, num_chosen, C)]
            grid_data.cls_prob: [num_lvl, (num_negs_all_img, C)]
            gt_labels: [B, (num_gt, )]
        Returns:
        """
        # split data by each img => [B, num_lvl, (num_gts, num_refine, num_chosen, C)]
        bag_data_list, grid_data_list = bag_data.split_each_img(), grid_data.split_each_img()
        # cascade_refine_data_list = cascade_refine_data.split_each_img()

        if gt_true_bboxes is None:
            gt_true_bboxes = [None] * len(gt_labels)
        if not_refine is None:
            not_refine = [None] * len(gt_labels)
        if cascade_refine_data is None:
            cascade_refine_data = [None] * len(gt_labels)

        refine_pts, refine_scores, not_refine, center_pts, chosen_pts, refine_geos = multi_apply(
            self.refine_single, bag_data_list, grid_data_list, gt_r_points, gt_labels, img_metas, gt_true_bboxes,
            not_refine, cascade_refine_data)

        if cascade_refine_data is not None:
            ret_cascade_data = CascadeData(refine_geos, refine_pts, chosen_pts)
        else:
            ret_cascade_data = None

        if self.debug_cfg['open']:
            for i, img_meta in enumerate(img_metas):
                refine_pts, center_pts, chosen_pts, gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine, refine_scores = \
                    refine_pts[i:i + 1], center_pts[i:i+1], chosen_pts[i:i + 1], gt_r_points[i:i + 1], gt_labels[i:i + 1], img_metas[
                                                                                                        i:i + 1], \
                    gt_true_bboxes[i:i + 1], not_refine[i:i + 1], refine_scores[i:i + 1]

                TestCPRHead.get(self).test_refine_point(refine_pts, None, center_pts, chosen_pts,
                                                        gt_r_points, gt_labels, img_metas, gt_true_bboxes, not_refine,
                                                        refine_scores,self.stage)
            # TestCPRHead.get(self).test_grid(grid_data_list, gt_labels, img_metas)
        return refine_pts, refine_scores, not_refine, ret_cascade_data


class PointRefiner2(PointRefiner):
    def refine_single(self, bag_data: PtAndFeat, grid_data: PtAndFeat, gt_r_points, gt_labels,
                      img_meta, gt_true_bboxes, not_refine=None, cascade_refine_data=None):
        """
        refine point in single image
        Args:
            bag_data.cls_prob: [lvl, (num_gts, num_refine, num_chosen, num_class)]
            bag_data.valid: [lvl, (num_gts, num_refine, num_chosen, 1)]
            grid_data.cls_prob: [lvl, (h, w, num_class)]
            grid_data.valid: [lvl, (h, w, num_class)]
            gt_labels: (num_gts,)
            gt_r_points: (num_gts, num_refine, 2)
        Returns:
        """
        bag_ins_prob = bag_data.ins_prob
        if bag_ins_prob is None:
            bag_ins_prob = [ins_out.softmax(dim=-2) for ins_out in bag_data.ins_outs]

        num_lvl = len(bag_ins_prob)
        num_gts, num_refine, num_chosen_lvl, _ = bag_ins_prob[0].shape
        pts_lvl = bag_ins_prob[0].new_tensor(range(num_lvl)).unsqueeze(dim=-1).repeat(1,
                                                                                      num_chosen_lvl).long().flatten()

        bag_cls_prob, bag_ins_prob = torch.cat(bag_data.cls_prob, dim=2), torch.cat(bag_ins_prob, dim=2)
        bag_pts, bag_valid = torch.cat(bag_data.pts, dim=2), torch.cat(bag_data.valid, dim=2)
        # grid_cls_prob = torch.cat(grid_data.cls_prob, dim=0)
        # grid_pts, grid_valid = torch.cat(grid_data.pts, dim=0), torch.cat(grid_data.valid, dim=0)

        if self.debug_cfg.get('ins_statistic', False):
            num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
            t_idx = torch.arange(num_gts).to(bag_ins_prob.device)
            bag_ins_score = bag_ins_prob[t_idx, :, :, gt_labels].flatten(0, 1)
            bag_cls_score = bag_cls_prob[t_idx, :, :, gt_labels].flatten(0, 1)
            idx = bag_cls_score.argsort(dim=1)
            idx0 = torch.arange(idx.shape[0]).unsqueeze(dim=1).repeat(1, idx.shape[1])
            self.statistic.print_log_mean_exp("ins_score", bag_ins_score[idx0, idx])

        # 2. split gt prob out, last one in bag
        gt_cls_prob = bag_cls_prob[..., -1:, :]
        gt_r_pts = bag_pts[..., -1:, :]
        assert (gt_r_pts[:, :, -1, :2] == gt_r_points[:,
                                          :1]).all(), f"last one of chosen dim should be annotation point."

        num_gts, num_refine, num_chosen, num_class = bag_cls_prob.shape
        gt_idx = torch.arange(len(gt_labels))
        merge_valid = bag_valid.reshape(num_gts, num_refine * num_chosen).bool()
        # 3. assign point
        if self.use_nearest_filter:
            merge_valid &= self.nearest_filter(bag_pts, gt_r_pts, gt_labels)
        if self.use_classify_filter:
            merge_valid &= self.classify_filter(bag_cls_prob, gt_labels)

        # 4. prob > th & prob > gt_prob * a
        bag_cls_prob = bag_cls_prob[gt_idx, ..., gt_labels].reshape(num_gts, num_refine * num_chosen)
        gt_cls_prob = gt_cls_prob[gt_idx, 0, ..., gt_labels].reshape(num_gts, 1)
        if self.debug_cfg.get('refine_statistic', False):
            self.statistic_scores(bag_cls_prob * merge_valid.float(), gt_cls_prob, num_lvl)
        # gt_cls_prob = gt_cls_prob.repeat(1, 1, num_chosen).flatten(1)
        merge_valid &= (bag_cls_prob > self.merge_th) & (bag_cls_prob > gt_cls_prob * self.gt_alpha)
        # graph filter
        # merge_valid &= self.graph_filter(bag_pts[..., -1:], merge_valid.reshape(num_gts, num_refine, num_chosen))
        # inside image filter
        merge_valid &= self.inside_img(bag_pts, img_meta['img_shape'])

        # 5. merge
        bag_pts = bag_pts.reshape(num_gts, num_refine * num_chosen, 3)
        bag_cls_prob = bag_cls_prob * merge_valid.float()
        bag_pts_weight = bag_cls_prob / (bag_cls_prob.sum(dim=1, keepdim=True) + 1e-8)
        refine_pts = (bag_pts[..., :2] * bag_pts_weight.unsqueeze(dim=-1)).sum(dim=1)  # (num_gts, 2)

        refine_scores = bag_cls_prob.sum(dim=-1) / ((bag_cls_prob > 0).float().sum(dim=-1) + 1e-8)  # (num_gts, )
        cur_not_refine = (refine_scores < self.refine_th)  # to prevent number of chosen points is 0
        not_refine = cur_not_refine if not_refine is None else not_refine | cur_not_refine
        refine_pts[not_refine] = gt_r_points[:, 0][not_refine]

        if self.return_score_type == 'max':
            refine_scores = bag_cls_prob.max(dim=-1)[0]
            refine_scores[refine_scores == 0] = self.refine_th / 2
        elif self.return_score_type == 'mean':
            pass
        else:
            raise ValueError

        # shape: [len_gt, (K, 3)]
        chosen_pts = [bag_pts[gt_i][chosen] for gt_i, chosen in enumerate(bag_pts_weight > 0)]
        self.statistic_after_refine(bag_pts_weight, bag_cls_prob, chosen_pts, bag_pts, not_refine, pts_lvl, num_lvl)
        return refine_pts, refine_scores, not_refine, chosen_pts, self.get_geo_output(chosen_pts, refine_pts)
