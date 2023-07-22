import matplotlib.pyplot as plt
import torch
import numpy as np
from .utils import Statistic


class TestCPRHead(object):
    DO_TEST = False
    DBI = dict(epoch=12, COUNT=-1, show=True, stage=-1)

    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    _INSTANCE = {}

    @staticmethod
    def get(obj):
        # assert isinstance(obj, PointRefiner)
        if obj not in TestCPRHead._INSTANCE:
            TestCPRHead._INSTANCE[obj] = test_cpr = TestCPRHead(len(TestCPRHead._INSTANCE))
            return test_cpr
        return TestCPRHead._INSTANCE[obj]

    def __init__(self, id):
        self.id = id
        self.sta = Statistic()
        self.count = 0

    def test_extract_point_feat(self, pts_in_feat, feat, pts_feats):
        if not TestCPRHead.DO_TEST:
            return
        pts_in_feat = pts_in_feat.reshape(-1, 2)
        x, y = pts_in_feat[:, 0], pts_in_feat[:, 1]
        idx_pts_feat = feat[0][:, y.long(), x.long()].permute(1, 0)
        pts_feats = pts_feats.reshape(-1, pts_feats.shape[-1])
        s = (idx_pts_feat - pts_feats).abs().max()
        if s > 1e-4:
            print("[test_extract_point_feat]:", s, x, y, feat.shape)

    def test_chosen_fpn_level(self, prob_bag_cls, prob_bag_ins, lvl_max_prob_idx,
                              all_gt_idx, all_gt_labels, num_all_gts, num_level):
        if not TestCPRHead.DO_TEST:
            return
        if num_level == 1:
            return
        sta = self.sta

        prob_bag_cls_l = prob_bag_cls[all_gt_idx, :, all_gt_labels]  # (num_gt, num_samples_all_lvl)
        prob_bag_ins_l = prob_bag_ins[all_gt_idx, :, all_gt_labels]
        for k, d in enumerate([prob_bag_cls_l, prob_bag_ins_l, (prob_bag_cls_l * prob_bag_ins_l)]):
            d = d.reshape(num_all_gts, num_level, -1)
            other_lvl_d = []
            chose_lvl_d = []
            for i in range(num_all_gts):
                for j in range(num_level):
                    if j == lvl_max_prob_idx[i]:
                        chose_lvl_d.append(d[i, j])
                    else:
                        other_lvl_d.append(d[i, j])
            chose_lvl_d, other_lvl_d = torch.stack(chose_lvl_d), torch.stack(other_lvl_d)
            print(f'{k} lvl data mean %.1e, %.1e, %.1e, %.1e, [%.1e, %.1e, %.1e, %.1e]' % ((
                                                                                               other_lvl_d.mean().item(),
                                                                                               chose_lvl_d.mean().item(),
                                                                                               chose_lvl_d.mean().item() / other_lvl_d.mean().item(),
                                                                                               (
                                                                                                       chose_lvl_d.max() / chose_lvl_d.mean()).item()) +
                                                                                           tuple(d.mean(dim=(0,
                                                                                                             2)).detach().cpu().numpy().tolist())))

        assert len(prob_bag_cls.shape) == 3
        prob_bag_cls_max, max_cls_idx = prob_bag_cls_l.max(dim=-1)
        prob_bag_ins_max, max_ins_idx = prob_bag_ins_l.max(dim=-1)
        print("max_i(cls)==max_i(ins)", sta.mean('max_i(cls)==max_i(ins)', (max_cls_idx == max_ins_idx).float()).item())
        rank = (prob_bag_ins_l >= prob_bag_ins_l[all_gt_idx, max_cls_idx].unsqueeze(dim=1)).float().sum(dim=1)
        print("ins rank of cls max", sta.mean('ins rank of cls max', rank).item())
        rank = (prob_bag_cls_l >= prob_bag_cls_l[all_gt_idx, max_ins_idx].unsqueeze(dim=1)).float().sum(dim=1)
        print("cls rank of ins max", sta.mean('cls rank of ins max', rank).item())
        x = prob_bag_cls_l[all_gt_idx, max_ins_idx] / prob_bag_cls_max
        print("cls[ins_max_idx] / cls_max", sta.mean('cls[ins_max_idx] / cls_max', x).item())
        x = prob_bag_ins_l[all_gt_idx, max_cls_idx] / prob_bag_ins_max
        print("ins[cls_max_idx] / ins_max", sta.mean('ins[cls_max_idx] / ins_max', x).item())

    def test_refine_point2(self, points, gt_true_bboxes, not_refine):
        points, gt_true_bboxes, not_refine = torch.cat(points), torch.cat(gt_true_bboxes), torch.cat(not_refine)
        inside = (gt_true_bboxes[:, 0] < points[:, 0]) & (points[:, 0] < gt_true_bboxes[:, 2]) & \
                 (gt_true_bboxes[:, 1] < points[:, 1]) & (points[:, 1] < gt_true_bboxes[:, 3])
        outside = inside.logical_not()
        outside_bboxes = gt_true_bboxes[outside]
        outside_size = ((outside_bboxes[:, 2] - outside_bboxes[:, 0]) * (
                outside_bboxes[:, 3] - outside_bboxes[:, 1])) ** 0.5
        sta = self.sta
        print("id", self.id)
        print("refine rate", sta.mean("refine rate", not_refine.logical_not().float()).item())
        print("outside rate", sta.mean("outside rate", outside.float()).item(),
              "outside size", sta.mean("outside size", outside_size).item())
        print()

    def test_grid(self, grid_data_list, gt_labels, img_metas):
        def pad_img(img, pad_shape):
            pad_img = np.zeros(pad_shape).astype(img.dtype)
            pad_img[:img.shape[0], :img.shape[1]] = img
            return pad_img

        def mask_img(mask, img):
            # return mask
            cmap = plt.get_cmap('jet')
            h, w = img.shape[:2]
            heatmap = Image.fromarray((cmap(mask)[..., :3] * 255).astype(np.uint8))
            heatmap = np.array(heatmap.resize((w, h))) / 255
            return (heatmap + img) / 2

        def plt_heatmap(grid_cls_prob, pos_labels, neg_labels, img, pad_shape):
            # grid_cls_prob[0, 0] = grid_cls_prob.max()
            # grid_cls_prob[-1, -1] = grid_cls_prob.min()
            img = np.array(img).astype(np.float32) / 255
            img = pad_img(img, pad_shape)

            k = 3
            plt.figure(figsize=(12, 6))
            for i, l in enumerate(pos_labels[:k]):
                plt.subplot(2, k, i + 1)
                plt.imshow(mask_img(grid_cls_prob[:, :, l], img))
                max_score = grid_cls_prob[:, :, l].max().round(2)
                plt.title(f"pos: {l}({TestCPRHead.CLASSES[l]}); max_score: {str(max_score)}")
            for i, l in enumerate(neg_labels[:k]):
                plt.subplot(2, k, k + i + 1)
                plt.imshow(mask_img(grid_cls_prob[:, :, l], img))
                max_score = grid_cls_prob[:, :, l].max().round(2)
                plt.title(f"neg: {l}({TestCPRHead.CLASSES[l]}); max_score: {str(max_score)}")
            plt.show()

        import matplotlib.pyplot as plt
        from PIL import Image
        grid_cls_prob = grid_data_list[0].cls_prob[0].cpu().numpy()
        grid_valid = grid_data_list[0].valid[0].float().cpu().numpy()
        labels = gt_labels[0].cpu().numpy()
        img_meta = img_metas[0]

        pos_labels = set(labels.tolist())
        neg_labels = list(set(list(range(80))) - pos_labels)
        from random import shuffle
        shuffle(neg_labels)
        pos_labels = list(pos_labels)

        img = Image.open(img_meta['filename'])
        plt_heatmap(grid_cls_prob, pos_labels, neg_labels, img, img_meta['pad_shape'])
        plt_heatmap(grid_valid, pos_labels, neg_labels, img, img_meta['pad_shape'])

    def save_cpr_data(self, img_meta, **kwargs):
        img_path = img_meta['filename']
        import os
        if not os.path.exists(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/'):
            os.makedirs(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/')
        img_name = os.path.split(img_path)[-1]
        np.savez(f'exp/debug/CPRData_e{TestCPRHead.DBI["epoch"]}/' + img_name + '.npz', dict=kwargs)

    def test_refine_point(self, points, chosen_lvl, center_pts_all, chosen_pts_all, gt_r_points, gt_labels, img_metas, gt_true_bboxes,
                          not_refine, fmt_points_score,stage):
        if not TestCPRHead.DO_TEST:
            return
        self.count += 1
        # print(TestCPRHead.DBI['COUNT'])
        if 0 <= TestCPRHead.DBI['COUNT'] < self.count:
            exit(-1)

        def to_numpy(data):
            data = data[0]
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data[0], torch.Tensor):
                return [d.detach().cpu().numpy() for d in data]

        def inside(points, gt_true_bboxes):
            is_inside = (gt_true_bboxes[:, 0] < points[:, 0]) & (points[:, 0] < gt_true_bboxes[:, 2]) & \
                        (gt_true_bboxes[:, 1] < points[:, 1]) & (points[:, 1] < gt_true_bboxes[:, 3])
            return is_inside

        def plot_img(img_meta):
            img_path = img_meta['filename']
            sw, sh = img_meta['scale_factor'][:2]

            img = Image.open(img_path)
            if 'corner' in img_meta:
                img = img.crop(img_meta['corner'])
            w, h = img.width, img.height
            print(w, h)
            img = img.resize((round(int(w * sw)), int(round(h * sh))))
            img = np.array(img)
            plt.figure(figsize=(14, 8))
            fig = set_plt(plt)
            plt.imshow(img)

        def plot_points(annotated_points, center_pts, chosen_points, refined_points, category_anns, img_true_bboxes=None):
            # colors = get_hsv_colors(80)
            colors = [(0.65, 0.65, 0.65)] * 80

            if chosen_points is not None:
                for i in range(len(center_pts)):
                    # link center_pt -- chosen_pts
                    for j in range(len(chosen_points[i])):
                        p1, p2 = center_pts[i], chosen_points[i][j]
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], ':', color=colors[category_anns[i]])  #
                    # link ann_pt -- refine_pt
                    plt.plot([annotated_points[i, 0], refined_points[i, 0]],
                             [annotated_points[i, 1], refined_points[i, 1]],
                             '--', linewidth=3, color=colors[category_anns[i]])
                    # link ann_pt -- center_pt
                    plt.plot([annotated_points[i, 0], center_pts[i, 0]],
                             [annotated_points[i, 1], center_pts[i, 1]],
                             '--', linewidth=3, color=colors[category_anns[i]])
                    plt.scatter(chosen_points[i][:, 0], chosen_points[i][:, 1], s=4, c='#fa2338', zorder=11)  #

            # 1. center point
            plt.scatter(center_pts[:, 0], center_pts[:, 1], s=12, c='#fa2338', zorder=11)
            # 2. ann_pt
            # plt.scatter(annotated_points[:, 0], annotated_points[:, 1], s=120, c='#22fe61', zorder=11)
            plt.scatter(annotated_points[:, 0], annotated_points[:, 1], s=12, c='#22fe61', zorder=11)
            # 3. refined_pt
            # plt.scatter(refined_points[:, 0], refined_points[:, 1], s=120, c='#ffff00', zorder=11)  # yellow
            plt.scatter(refined_points[:, 0], refined_points[:, 1], s=12, c='#ffff00', zorder=11)  # yellow

            # is_inside = inside(refined_points, img_true_bboxes)
            # plt.scatter(refined_points[is_inside, 0], refined_points[is_inside, 1], s=120, c='#ffff00', zorder=11) # yellow
            # plt.scatter(refined_points[np.logical_not(is_inside), 0], refined_points[np.logical_not(is_inside), 1],
            #             s=40, c=(0, 0, 0))

        def plot_scores(refined_points, img_scores):
            if img_scores is not None:
                for i in range(len(refined_points)):
                    plt.text(refined_points[i][0], refined_points[i][1], s=f"{(img_scores[i] * 100).round(2)}",
                             color=(1, 1, 1), fontsize=10)

        self.test_refine_point2(points, gt_true_bboxes, not_refine)

        img_path = img_metas[0]['filename']
        img_gt_r_points = to_numpy(gt_r_points)
        img_points = to_numpy(points)
        img_gt_labels = to_numpy(gt_labels)
        img_true_bboxes = to_numpy(gt_true_bboxes) if gt_true_bboxes is not None else None
        img_center_pts = to_numpy(center_pts_all)
        img_chosen_pts = to_numpy(chosen_pts_all)
        img_chosen_lvl = to_numpy(chosen_lvl) if chosen_lvl is not None else None
        img_not_refine = to_numpy(not_refine) if not_refine is not None else None
        img_scores = to_numpy(fmt_points_score) if fmt_points_score is not None else None

        img_gt_points = img_gt_r_points[:, 0]
        if img_true_bboxes is not None:
            assert len(img_true_bboxes) == len(img_gt_points), f"{len(img_true_bboxes)} vs {len(img_gt_points)}"
        img_center_pts = img_center_pts[:, 0]

        from PIL import Image
        import matplotlib.pyplot as plt
        from ssdcv.vis.visualize import get_hsv_colors, draw_a_bbox
        from ssdcv.plot_paper.plt_paper_config import set_plt
        import os

        if not os.path.exists(f"exp/debug/ccpr_paper/CCPR_3stage_var/stage_{stage}"):
            os.makedirs(f"exp/debug/ccpr_paper/CCPR_3stage_var/stage_{stage}")

        annotated_points, center_pts, chosen_points, refined_points = img_gt_points, img_center_pts, img_chosen_pts, img_points

        print(img_metas)

        plot_img(img_metas[0])
        plot_points(annotated_points, center_pts, chosen_points, refined_points, img_gt_labels, img_true_bboxes)
        # plot_scores(refined_points, img_scores)
        print(plt)
        img_name = os.path.split(img_path)[-1]
        plt.savefig(f"exp/debug/ccpr_paper/CCPR_3stage_var/stage_{stage}/vis_s{stage}_{img_name}")
        if TestCPRHead.DBI['show']:
            plt.show()
        else:
            plt.clf()

        # plot_img(img_metas[0])
        # plot_points(annotated_points, None, refined_points, img_gt_labels, img_true_bboxes)
        # # plot_scores(refined_points, img_scores)
        # img_name = os.path.split(img_path)[-1]
        # plt.savefig("exp/debug/CPR/vis2_{}".format(img_name))
        # if TestCPRHead.DBI['show']:
        #     plt.show()
        # else:
        #     plt.clf()

    def show_scores(self, bag_data, gt_labels, img_meta):
        # (num_gts, num_refine, num_lvl, num_chosen, num_class)
        bag_cls_prob = torch.stack(bag_data.cls_prob, dim=2)
        bag_pts, bag_valid = torch.stack(bag_data.pts, dim=2), torch.stack(bag_data.valid, dim=2)
        bag_cls_prob = bag_cls_prob * bag_valid
        gt_cls_prob = bag_cls_prob[:, :, :, -1]
        gt_pts = bag_pts[:, :, :, -1:]

        num_gts, num_refine, num_lvl, num_chosen, num_class = bag_cls_prob.shape

        gt_idx = torch.arange(len(gt_labels))
        gt_cls_prob = gt_cls_prob[gt_idx, ..., gt_labels]
        bag_cls_prob = bag_cls_prob[gt_idx, ..., gt_labels]
        # norm_bag_cls_prob = bag_cls_prob / bag_cls_prob.max(dim=-1, keepdim=True)[0]
        # norm_gt_cls_prob = gt_cls_prob / bag_cls_prob.max(dim=-1)[0]

        bag_cls_prob_cpu = bag_cls_prob.reshape(num_gts * num_refine, num_lvl, num_chosen).cpu().numpy()
        gt_cls_prob_cpu = gt_cls_prob.reshape(num_gts * num_refine, num_lvl).cpu().numpy()

        dis = (bag_pts[..., :2] - gt_pts[..., :2])
        dis = (dis ** 2).sum(dim=-1) ** 0.5
        dis = dis.reshape(num_gts * num_refine, num_lvl, num_chosen).cpu().numpy()

        import matplotlib.pyplot as plt
        for gt_i in range(num_gts):
            self.plot_img(img_meta)
            self.plt_point(gt_pts.flatten(0, 1).cpu().numpy()[gt_i:gt_i + 1, 0, 0],
                           bag_pts.flatten(2, 3).flatten(0, 1).cpu().numpy()[gt_i:gt_i + 1])
            plt.show()

            for i in range(num_lvl):
                plt.subplot(2, 2, i + 1)
                s = bag_cls_prob_cpu[gt_i, i]
                plt.scatter(dis[gt_i, i][s > 0], s[s > 0], s=3)
                plt.scatter([0.0], gt_cls_prob_cpu[gt_i, i], s=3, c='r')
            plt.show()

    # tool function
    def plot_img(self, img_meta):
        import matplotlib.pyplot as plt
        from ssdcv.plot_paper.plt_paper_config import set_plt
        img = TestCPRHead.load_img(img_meta)
        plt.figure(figsize=(14, 8))
        fig = set_plt(plt)
        plt.imshow(img)

    def plt_point(self, annotated_points, chosen_points):
        import matplotlib.pyplot as plt
        for i in range(len(chosen_points)):
            plt.scatter(chosen_points[i][:, 0], chosen_points[i][:, 1], s=4, c='#fa2338', zorder=11)  #
        plt.scatter(annotated_points[:, 0], annotated_points[:, 1], s=12, c='#22fe61', zorder=11)

    @staticmethod
    def load_img(img_meta):
        from PIL import Image
        import numpy as np

        img_path = img_meta['filename']
        sw, sh = img_meta['scale_factor'][:2]

        img = Image.open(img_path)
        if 'corner' in img_meta:
            img = img.crop(img_meta['corner'])
        w, h = img.width, img.height
        print(w, h)
        img = img.resize((round(int(w * sw)), int(round(h * sh))))
        img = np.array(img)
        return img

    @staticmethod
    def load_pad_img(img_meta, pad_HW=None):
        def pad_img(img, pad_shape):
            pad_img = np.zeros(pad_shape).astype(img.dtype)
            pad_img[:img.shape[0], :img.shape[1]] = img
            return pad_img

        img = TestCPRHead.load_img(img_meta)
        img = img.astype(np.float32) / 255

        if pad_HW is None:
            pad_HW = img_meta['pad_shape']
        elif len(pad_HW) < len(img_meta['pad_shape']):
            pad_HW = tuple(list(pad_HW) + list(img_meta['pad_shape'])[len(pad_HW):])
        img = pad_img(img, pad_HW)
        return img

    @staticmethod
    def show_bbox(gt_true_bbox):
        from ssdcv.vis.visualize import draw_bbox
        draw_bbox(plt.gca(), gt_true_bbox, normalized_label=False)

    @staticmethod
    def mask_img(mask, img):
        from PIL import Image
        import matplotlib.pyplot as plt
        # return mask
        cmap = plt.get_cmap('jet')
        h, w = img.shape[:2]
        heatmap = Image.fromarray((cmap(mask)[..., :3] * 255).astype(np.uint8))
        heatmap = np.array(heatmap.resize((w, h))) / 255
        return (heatmap + img) / 2

    @staticmethod
    def show_feats(feats, img_metas):
        import matplotlib.pyplot as plt
        pad_img = TestCPRHead.load_pad_img(img_metas[0])
        mask_img = TestCPRHead.mask_img

        num_lvl = len(feats)
        i = 0
        feat_c = feats[0].shape[1]
        rand_c = torch.randint(feat_c, (1,))[0]
        min_v = min([feat.min() for feat in feats])
        max_v = max([feat.max() for feat in feats])
        feats = [(feat - min_v) / (max_v - min_v) for feat in feats]
        # for feat in feats:
        #     plt.subplot(num_lvl, 3, 3 * i + 1)
        #     f = feat[0, 0].detach().cpu().numpy()
        #     plt.imshow(mask_img(f, pad_img))
        #     plt.subplot(num_lvl, 3, 3 * i + 2)
        #     f = feat[0].mean(dim=0).detach().cpu().numpy()
        #     plt.imshow(mask_img(f, pad_img))
        #     plt.subplot(num_lvl, 3, 3 * i + 3)
        #     f = feat[0, rand_c].detach().cpu().numpy()
        #     plt.imshow(mask_img(f, pad_img))
        #     i += 1

        channels = [0, 50, 100, 150, 200, 250]
        for feat in feats:
            for j, c in enumerate(channels):
                plt.subplot(num_lvl, len(channels), len(channels) * i + j + 1)
                f = feat[0, c].detach().cpu().numpy()
                plt.imshow(mask_img(f, pad_img))
            i += 1
        plt.show()
