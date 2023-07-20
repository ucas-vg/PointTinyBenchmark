import numpy as np
from huicv.vis.visualize import draw_a_bbox
import matplotlib.pyplot as plt


class ResultFilter(object):
    def __init__(self, num_per_gt, score_th=0.2):
        self.num_per_gt = num_per_gt
        self.score_th = score_th

    def __call__(self, anns, result):
        len_ann = len(anns)
        # len_ann = len([ann for ann in anns if not ann['ignore']])
        result = result[result[:, 4] > self.score_th]
        if self.num_per_gt > 0 and len(result) > len_ann * self.num_per_gt:
            result = np.array(sorted(result, key=lambda x: -x[4]))[:len_ann * self.num_per_gt]
        return result


def norm(x):
    return (x * x).sum(axis=-1) ** 0.5


class MinDisCluster(object):
    def __call__(self, anns, result):
        gt_pts = np.array([ann['point'] for ann in anns])
        cxs = (result[:, 0] + result[:, 2]) / 2
        cys = (result[:, 1] + result[:, 3]) / 2
        det_center = np.stack((cxs, cys), axis=-1)
        det_center = det_center.reshape((1, -1, 2))
        gt_pts = gt_pts.reshape((-1, 1, 2))
        dis = norm(gt_pts - det_center)
        C = np.argmin(dis, axis=0)
        clusters = [[] for _ in range(len(gt_pts))]
        for i, c in enumerate(C):
            clusters[c].append(i)
        return clusters


class MinIOUCluster(object):
    def __call__(self, anns, result):
        gt_bbox = np.array([ann['bbox'] for ann in anns])


class TopkFilter(object):
    def __init__(self, k):
        self.k = k

    def __call__(self, ann, clu_res):
        if len(clu_res) > self.k:
            clu_res = np.array(sorted(clu_res, key=lambda x: -x[4]))[:self.k]
        return clu_res


class ContainPointFilter(object):
    def __call__(self, ann, clu_res):
        _, _, w, h = ann['bbox']
        px, py = ann['point']
        keep = (px < clu_res[:, 2]) & (px >= clu_res[:, 0]) & (py < clu_res[:, 3]) & (py >= clu_res[:, 1])
        return clu_res[keep]


class GrowRateFilter(object):
    def __init__(self, max_grow_rate):
        self.max_grow_rate = max_grow_rate

    def __call__(self, ann, clu_res):
        _, _, w, h = ann['bbox']
        px, py = ann['point']
        max_w = w * self.max_grow_rate
        max_h = w * self.max_grow_rate
        keep = (np.abs(clu_res[:, [0, 2]] - px) < max_w) & (np.abs(clu_res[:, [1, 3]] - py) < max_h)
        keep = np.logical_and(keep[:, 0], keep[:, 1])
        return clu_res[keep]


class ClusterFilters(object):
    def __init__(self, cluster_filters):
        self.cluster_filters = cluster_filters

    def __call__(self, anns, result, clusters):
        for idx, cluster in enumerate(clusters):
            clu_res = result[cluster]
            if len(clu_res) > 0:
                for clu_filter in self.cluster_filters:
                    clu_res = clu_filter(anns[idx], clu_res)
            clusters[idx] = clu_res
        return clusters


class SimpleBoxGenerator(object):
    def __init__(self, mode='minmax', include_point=True):
        self.mode = mode
        self.include_point = include_point

    def __call__(self, anns, result, clusters):
        boxes, idxes = [], []
        for idx, clu_res in enumerate(clusters):
            if len(clu_res) == 0: continue
            if self.mode == 'center':
                if len(clu_res) > 1:
                    x = clu_res[:, [0, 2]].mean(axis=-1)
                    y = clu_res[:, [1, 3]].mean(axis=-1)
                    x1, x2 = x.min(), x.max()
                    y1, y2 = y.min(), y.max()
                else:
                    x1, y1, x2, y2 = clu_res[0, :4]
            elif self.mode == 'mean':
                x1, y1, x2, y2 = clu_res[:, :4].mean(axis=0)
            elif self.mode == 'weight_mean':
                x1, y1, x2, y2 = (clu_res[:, :4] * clu_res[:, [4]]).sum(axis=0) / clu_res[:, 4].sum()
            elif self.mode == 'minmax':
                x1 = clu_res[:, [0]].min()
                y1 = clu_res[:, [1]].max()
                x2 = clu_res[:, [2]].min()
                y2 = clu_res[:, [3]].max()
            else:
                raise ValueError('')

            if self.include_point:
                ann = anns[idx]['point']
                x1, y1 = min(ann[0], x1), min(ann[1], y1)
                x2, y2 = max(ann[0], x2), max(ann[1], y2)
            boxes.append([x1, y1, x2, y2])
            idxes.append(idx)
        return np.array(boxes), np.array(idxes)


def show_annos(annos, keys=['point', 'bbox', 'true_bbox'], color=None, dash_bbox=False, linewidth=2):
    for ann in annos:
        if 'point' in ann and 'point' in keys:
            xc, yc = ann['point']
            plt.scatter(xc, yc, color='b' if color is None else color, s=20)
        if 'true_bbox' in ann and 'true_bbox' in keys:
            x1, y1, w, h = ann['true_bbox']
            draw_a_bbox([x1, y1, x1+w, y1+h], 'lime' if color is None else color, dash=dash_bbox, fill=True)
        if 'bbox' in ann and 'bbox' in keys:
            x1, y1, w, h = ann['bbox']
            draw_a_bbox([x1, y1, x1+w, y1+h], 'b' if color is None else color, linewidth, dash=dash_bbox)


def show_cluster(anns, clusters, idx, color=None, ann_kwargs={}, show_ann=True, show_center=False, show_bbox=True):
    dets = clusters[idx]
    if len(dets) == 0:
        print('no bbox')
        return
    color = (1, 1, 1) if color is None else color
    print("cluster:", len(dets))
    for det in dets:
        if show_bbox:
            draw_a_bbox(det[:4], color)
        if show_center:
            x1, y1, x2, y2 = det[:4]
            plt.scatter((x1+x2)/2, (y1+y2)/2, color=color, s=120*det[-1]*det[-1], alpha=0.7)
    if show_ann:
        show_annos(anns[idx:idx+1], **ann_kwargs)
    # plt.axes().add_patch(draw_a_bbox(cluster_boxes[idx][:4], 'r'))
    # plt.scatter(*ann['point'], color='b', s=10)
