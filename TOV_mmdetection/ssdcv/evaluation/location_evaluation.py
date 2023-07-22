import numpy as np
from copy import deepcopy
import sys

# python ssdcv/evaluation/location_evaluation.py \
# '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json' \
# exp//latest_result.json --matchThs 1.0

# part1: matcher, to get matched gt/ignored gt of each det result start ###############################
BACKGROUND = -1


class GTMatcher(object):
    """
        1. if a det have multi regular gts that V[det, gt] > v_th, choose the max V gt as matched;
        2. if multi dets match to same regular gt, set the max score det as matched with the gt,
            and the lower score det try to match other gt
        3. if a det without regular gt matched after 1,2, try to match ignored gt and if matched set it as ignore det.
    """

    def __init__(self, eps=1e-8, LOG=None):
        self.eps = eps
        self.LOG = LOG

    def cal_value(self, D, G):
        """
            for bbox detection, cal_value is calculate IOU of dets and gts
        """
        raise NotImplementedError()

    def _match_to_regluar_gt_no_repeat(self, V, v_th, M):
        """
            we assume the input det result have been descend sorted by score,
            this function match det to a gt which have not been matched before and have max value with the det and IOU[det, gt]>v_th.
            args:
                V: value of det box with gt box,for bbox detection it is IOU, shape = (len(D), len(G)), V range in [0, 1]
                v_th: the threshold for macth
            return:
                M: matched gt id for each det, shape=len(D)
        """
        left_g = V.shape[1]
        keep = np.array([1] * V.shape[1])
        for i in range(V.shape[0]):
            if left_g <= 0: continue
            j = np.argmax(V[i, :] * keep)
            if V[i, j] >= v_th:
                M[i] = j
                keep[j] = 0  # remove gt j
                left_g -= 1

    def _match_to_regluar_gt_no_repeat_v2(self, V, v_th, scores, M):
        equal_score_edge = [0]
        last_score = scores[0]
        for i in range(1, len(scores)):
            if abs(scores[i] - last_score) < self.eps:
                continue
            equal_score_edge.append(i)
            last_score = scores[i]
        equal_score_edge.append(len(scores))

        keep_det = np.array([1] * V.shape[0]).reshape((-1, 1))
        keep_gt = np.array([1] * V.shape[1]).reshape((1, -1))
        left_gt = V.shape[1]
        for i in range(len(equal_score_edge) - 1):
            if left_gt == 0: break
            s, e = equal_score_edge[i], equal_score_edge[i + 1]
            for _ in range(s, e):
                if left_gt == 0: break
                idx = np.argmax(((V[s:e] * keep_det[s:e]) * keep_gt).reshape((-1,)))
                det_id = s + idx // V.shape[1]
                gt_id = idx % V.shape[1]

                if V[det_id, gt_id] >= v_th:
                    M[det_id] = gt_id
                    keep_det[det_id] = 0
                    keep_gt[:, gt_id] = 0
                    left_gt -= 1

    def _match_as_ignore_det(self, V, v_th, start_gt_idx, M, ID):
        """
            args:
                V: value of det box with gt box,for bbox detection it is IOU, shape = (len(D), len(G)), where
                v_th: the threshold for macth
            return:
                M: matched gt id for each det, shape=len(D)
                ID: whether a det is a ignored det, shape=len(D),
                    ignore det will calculate as neither TP(true positive) nor FP(false positive)
        """
        for i in range(V.shape[0]):
            if M[i] != BACKGROUND: continue
            j = np.argmax(V[i, :])
            if V[i, j] >= v_th:
                M[i] = j + start_gt_idx
                ID[i] = True

    def __call__(self, D, det_scores, G, IG, v_th, multi_match_not_false_alarm, multi_match_v_th=None):
        """
            D must be sorted by det_scores
        """
        if multi_match_v_th is None:
            multi_match_v_th = v_th

        M = np.array([BACKGROUND] * len(D))
        ID = np.array([False] * len(D))  # ignore det

        if len(G) > 0:
            V = self.cal_value(D, G)
            if self.LOG is not None: print('V(D, G):\n', V, file=self.LOG)
            # match det to regular gt with no repeated
            # self._match_to_regluar_gt_no_repeat(V, v_th, M)
            self._match_to_regluar_gt_no_repeat_v2(V, v_th, det_scores, M)

        if len(IG) > 0:
            IV = self.cal_value(D, IG)
            if self.LOG is not None: print('V(D, IG):\n', IV, file=self.LOG)
            # match det to ignore gt with repeated
            self._match_as_ignore_det(IV, v_th, len(G), M, ID)

        if multi_match_not_false_alarm and len(G) > 0:
            # if do not treat multi det that match same gt as false alarm, set them as ignore det
            self._match_as_ignore_det(V, multi_match_v_th, 0, M, ID)
        return M, ID, det_scores


# class BoxMatcher(GTMatcher):
#     def cal_value(self, dets, gts):
#         return IOU(dets, gts)


class PointMatcher(GTMatcher):
    def cal_value(self, dets, gts):
        """
            L2 distance matcher for (xc, yc) det and (xc, yc, w, h) gt.
            return:
                square_of_distance = (dx*dx + dy*dy)
                # add 1 to avoid divide by 0, transform range of V to (0, 1], like IOU
                V = 1/(square_of_distance+1)
                return V
        """
        det_values = np.empty((len(dets), len(gts)))
        for i in range(len(dets)):
            d = (dets[i].reshape((1, -1)) - gts[:, :2]) / gts[:, 2:]
            det_values[i, :] = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        return 1 / (1 + det_values)

    def __call__(self, dets, det_scores, gts, ignore_gts, dis_th, multi_match_not_false_alarm, multi_match_dis_th=None):
        v_th = 1 / (dis_th * dis_th + 1)
        multi_match_v_th = 1 / (multi_match_dis_th * multi_match_dis_th + 1) if multi_match_dis_th is not None else v_th
        if self.LOG: print('v_th:', v_th, file=self.LOG)
        return super(PointMatcher, self).__call__(
            dets, det_scores, gts, ignore_gts, v_th, multi_match_not_false_alarm, multi_match_v_th
        )
# part1: matcher, to get matched gt/ignored gt of each det result end ###############################


# part2: recall precision cal, to get recall and precision from match result start ###############################
def cal_recall_precision(match_gts, dets_score, len_pos):
    idx = np.argsort(-dets_score)
    match_gts, dets_score = match_gts[idx], dets_score[idx]
    is_pos = (match_gts != BACKGROUND)
    TP = np.cumsum(is_pos.astype(np.float32))
    recall = TP / (len_pos + 1e-12)
    precision = TP / np.arange(1, len(is_pos) + 1)

    last_r = -1
    final_recall = []
    final_precison = []
    chosen_idx = []
    for i, (r, p) in enumerate(zip(recall, precision)):
        # for each recall choose the max precision
        if abs(last_r - r) < 1e-10:
            continue
        final_recall.append(r)
        final_precison.append(p)
        last_r = r
        chosen_idx.append(i)
    recall = np.array(final_recall)
    precision = np.array(final_precison)

    if len(recall) == 0:  # no det
        if len_pos == 0:  # no gt
            recall, precision = np.array([1.]), np.array([1.])
        else:  # have gt
            recall, precision = np.array([0]), np.array([0.])

    if LocationEvaluator.SAVE_RECALL_PRECISION_PATH is not None:
        np.savez(LocationEvaluator.SAVE_RECALL_PRECISION_PATH, recall=recall, precision=precision, dets_score=dets_score[chosen_idx])
    return recall, precision


def cat(arrays):
    if len(arrays) == 0:
        return np.array([])
    else:
        return np.concatenate(arrays)


def match_and_cal_recall_precision(all_dets, all_dets_score, all_gts, all_gts_ignore, match_th, maxDets,
                                  matcher, matcher_kwargs={}):
    """
        match and cal recall and precision of single condition, which means
            single class, single size_range, single match_th
        called by evaluate_in_multi_condition
    """
    assert (set(all_gts.keys()) | set(all_dets.keys())) == set(all_gts.keys()), "all det image must in gt"
    all_match_gts, all_sorted_dets_scores, all_dets_keep, len_pos = {}, {}, {}, 0
    for i in all_gts:
        gts, gts_ignore = all_gts[i], all_gts_ignore[i]
        dets, dets_score = all_dets[i], all_dets_score[i]

        len_pos += len(gts_ignore) - np.sum(gts_ignore)
        if len(dets) > 0:
            G, IG = gts[np.logical_not(gts_ignore)], gts[gts_ignore]
            # D = descend_sort_by_score(D)
            idx = np.argsort(-dets_score)
            dets, dets_score = dets[idx][:maxDets], dets_score[idx][:maxDets]

            match_gts, dets_ignore, dets_score = matcher(dets, dets_score, G, IG, match_th, **matcher_kwargs)
            all_match_gts[i] = match_gts
            all_sorted_dets_scores[i] = dets_score
            all_dets_keep[i] = np.logical_not(dets_ignore)

    # filter ignore det out when evaluate AP
    images_id = list(all_match_gts.keys())
    match_gts_array = cat([all_match_gts[img_id][all_dets_keep[img_id]] for img_id in images_id])
    dets_scores_array = cat([all_sorted_dets_scores[img_id][all_dets_keep[img_id]] for img_id in images_id])

    recall, precision = cal_recall_precision(match_gts_array, dets_scores_array, len_pos)
    return {"recall": recall, "precision": precision}


def match_and_cal_recall_precision_of_every_image(all_dets, all_dets_score, all_gts, all_gts_ignore,
                                                 match_th, maxDets, matcher,
                                                  matcher_kwargs={}):
    """
      debug function: it is a function to cal recall and precision for each image
      try to measure
    """
    assert (set(all_gts.keys()) | set(all_dets.keys())) == set(all_gts.keys()), "all det image must in gt"
    all_recall, all_precision = {}, {}
    for i in all_gts:  # for each image
        gts, gts_ignore = all_gts[i], all_gts_ignore[i]
        dets, dets_score = all_dets[i], all_dets_score[i]

        if len(dets) > 0:
            G, IG = gts[np.logical_not(gts_ignore)], gts[gts_ignore]
            # D = descend_sort_by_score(D)
            idx = np.argsort(-dets_score)
            dets, dets_score = dets[idx][:maxDets], dets_score[idx][:maxDets]

            match_gts, dets_ignore, dets_score = matcher(dets, dets_score, G, IG, match_th, **matcher_kwargs)
            dets_keep = np.logical_not(dets_ignore)

            if len(G) > 0 and (np.sum(dets_keep) == 0):
                # miss
                all_recall[i] = [0.]
                all_precision[i] = [-2]
            if len(G) == 0 and (np.sum(dets_keep) > 0):
                # flase alarm
                all_recall[i] = [0]
                all_precision[i] = [-3]
            elif len(G) == 0 and (np.sum(dets_keep) == 0):
                all_recall[i] = [1.]
                all_precision[i] = [2.]
            elif len(G) > 0 and np.sum(dets_keep) > 0:
                recall, precision = cal_recall_precision(match_gts[dets_keep], dets_score[dets_keep], len(G))
                all_recall[i] = recall
                all_precision[i] = precision
        elif len(G) > 0:
            # miss gt
            all_recall[i] = [0.]
            all_precision[i] = [-2]
        else:
            all_recall[i] = [1.]
            all_precision[i] = [1.]

    return {"all_recall": all_recall, "all_precision": all_precision}


def evaluate_in_multi_condition(all_dets, all_dets_score, all_gts, all_gts_ignore,
                                match_th_list, size_ranges, maxDets_list, matcher,
                                matcher_kwargs={}, evaluate_img_separate=False):
    """
        evaluate_img_seperate: if True, then for each image, calculate recall and precision, only for analysis
    """
    res = {
        'match_th_idx': [],
        'size_range_idx': [],
        'maxDets_idx': []
    }
    for si, (min_size, max_size) in enumerate(size_ranges):  # choose a size_range

        # set gt that size out of [min_size, max_size) as ignored gt
        all_gts_ignore_copy = deepcopy(all_gts_ignore)
        for i in all_gts_ignore_copy:
            gts = all_gts[i]
            if len(gts) <= 0: continue
            gts_ignore = all_gts_ignore_copy[i]
            sizes = np.sqrt((gts[:, -1] * gts[:, -2]))
            gts_ignore[np.logical_or(sizes >= max_size, sizes < min_size)] = True

        for mi, match_th in enumerate(match_th_list):
            for mdi, maxDets in enumerate(maxDets_list):
                if not evaluate_img_separate:
                    results = match_and_cal_recall_precision(
                        all_dets, all_dets_score, all_gts, all_gts_ignore_copy, match_th, maxDets,
                        matcher, matcher_kwargs)
                else:
                    results = match_and_cal_recall_precision_of_every_image(
                        all_dets, all_dets_score, all_gts, all_gts_ignore_copy, match_th, maxDets,
                        matcher, matcher_kwargs)
                res['match_th_idx'].append(mi)
                res['size_range_idx'].append(si)
                res['maxDets_idx'].append(mdi)
                for key, value in results.items():
                    if key not in res:
                        res[key] = [value]
                    else:
                        res[key].append(value)
    return res
# part2: recall precision cal, to get recall and precision from match result end ###############################


# part3: transform input to call evaluate_in_multi_condition start ###############################
def group_by(dicts, key):
    res = {}
    for objs in dicts:
        v = objs[key]
        if v in res:
            res[v].append(objs)
        else:
            res[v] = [objs]
    return res


def get_center_w_h(x, y, w, h):
    return [x + (w - 1) / 2, y + (h - 1) / 2, w, h]


class LocationEvaluator(object):
    """
    example:
    --------------------------------------------------------------------
        MAX_SIZE = 1e5
        evaluator = LocationEvaluator(
            areaRng=[(1**2, 20**2), (20**2, MAX_SIZE**2), (1**2, MAX_SIZE**2)],
            matchThs=[0.5, 1.0, 2.0],
            matcher_kwargs=dict(multi_match_not_false_alarm=False)
        )

        # first call way
        from pycocotools.coco import COCO
        gt_jd = COCO(gt_file)
        det_jd = gt_jd.loadRes(det_file)
        LocationEvaluator.add_center_from_bbox_if_no_point(det_jd)
        res = evaluator(det_jd, gt_jd)

        # second call way
        gt_jd = json.load(open(gt_file))
        det_jd = json.load(open(det_file))
        LocationEvaluator.add_center_from_bbox_if_no_point(det_jd)
        res = evaluator(det_jd, gt_jd)
    --------------------------------------------------------------------
    return:
    --------------------------------------------------------------------
    res[cate_idx] = {
        'match_th_idx': [....],
        'size_range)idx': [....],
        'maxDets_idx': [....],
        'recall': [[...], ....],
        'precision': [[...], ....]
        }
    category: gt_jd['categories'][cate_idx]
    """

    SAVE_RECALL_PRECISION_PATH = None

    def __init__(self, evaluate_img_separate=False, class_wise=False, use_ignore_attr=True,
                 location_param={}, matcher_kwargs=dict(multi_match_not_false_alarm=False), **kwargs):
        """
            evaluate_img_separate: if True, then for each image, calculate recall and precision, only set True for analysis
        """
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        if not class_wise:
            self.matchThs = [0.5, 1.0, 2.0]
            self.maxDets = [200] if "maxDets" not in kwargs else kwargs["maxDets"]
            self.areaRng = [[1 ** 2, 1e5 ** 2], [1 ** 2, 20 ** 2], [1 ** 2, 8 ** 2], [8 ** 2, 12 ** 2],
                            [12 ** 2, 20 ** 2], [20 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]] \
                if "areaRng" not in kwargs else kwargs["areaRng"]
            self.areaRngLbl = ['all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable'] \
                if "areaRngLbl" not in kwargs else kwargs["areaRngLbl"]
        else:
            self.matchThs = [1.0]
            self.maxDets = [200] if "maxDets" not in kwargs else kwargs["maxDets"]
            self.areaRng = [[1 ** 2, 1e5 ** 2]] if "areaRng" not in kwargs else kwargs["areaRng"]
            self.areaRngLbl = ['all'] if "areaRngLbl" not in kwargs else kwargs["areaRngLbl"]

        for key, value in location_param.items():
            assert key in ['maxDets', 'recThrs', 'matchThs', 'areaRng', 'areaRngLbl'], f"{key} is not valid"
            self.__setattr__(key, value)
        if isinstance(self.recThrs, str):
            self.recThrs = eval(self.recThrs)
        self.recThrs = np.array(self.recThrs)
        assert len(self.areaRng) == len(self.areaRngLbl)

        self.size_ranges = np.array([[min_area**0.5, max_area**0.5] for min_area, max_area in self.areaRng])

        self.class_wise = class_wise
        self.evaluate_img_separate = evaluate_img_separate
        self.use_ignore_attr = use_ignore_attr

        self.matcher = PointMatcher()
        self.matcher_kwargs = matcher_kwargs

        self.gt_jd = None

    def __call__(self, det_jd, gt_jd):
        try:
            from pycocotools.coco import COCO
            if isinstance(det_jd, COCO):
                det_jd = list(det_jd.anns.values())
            if isinstance(gt_jd, COCO):
                gt_jd = gt_jd.dataset
        except ModuleNotFoundError as e:
            pass
        self.gt_jd = gt_jd
        return self.evaluate_multi_class(det_jd, gt_jd)

    def evaluate_multi_class(self, det_jd, gt_jd):
        res_set = []
        for cate in gt_jd['categories']:
            gt_annos = [anno for anno in gt_jd['annotations'] if anno['category_id'] == cate['id']]
            single_class_det_jd = [det for det in det_jd if det['category_id'] == cate['id']]
            single_class_gt_jd = {key: value for key, value in gt_jd.items() if key != 'annotations'}
            single_class_gt_jd['annotations'] = gt_annos
            # print("****************************** evaluating on", cate, len(gt_annos), len(single_class_det_jd))
            res = self.evaluate_single_class(single_class_det_jd, single_class_gt_jd)
            res_set.append(res)
        return res_set

    def evaluate_single_class(self, det_jd, gt_jd):
        g_det_jd = {img['id']: [] for img in gt_jd['images']}
        g_det_jd.update(group_by(det_jd, "image_id"))

        g_gt_jd = {img['id']: [] for img in gt_jd['images']}
        g_gt_jd.update(group_by(gt_jd['annotations'], 'image_id'))

        # all_dets_bbox = {img_id: [det['bbox'] for det in dets] for img_id, dets in g_det_jd.items()}
        all_dets_point = {img_id: np.array([det['point'] for det in dets], dtype=np.float32) for img_id, dets in
                          g_det_jd.items()}
        all_dets_score = {img_id: np.array([det['score'] for det in dets], dtype=np.float32) for img_id, dets in
                          g_det_jd.items()}
        # all_gts_point = {img_id: np.array([get_center(*gt['bbox']) for gt in gts], dtype=np.float32)
        #  for img_id, gts in g_gt_jd.items()}
        # all_gts_score = {img_id: np.array([0.9 for det in dets], dtype=np.float32)
        #  for img_id, dets in g_gt_jd.items()}
        all_gts_centerwh = {img_id: np.array([get_center_w_h(*gt['bbox']) for gt in gts], dtype=np.float32) for
                            img_id, gts in g_gt_jd.items()}
        all_gts_ignore = {img_id:  self.get_ignore(gts) for img_id, gts in g_gt_jd.items()}

        res = evaluate_in_multi_condition(all_dets_point, all_dets_score, all_gts_centerwh, all_gts_ignore,
                                          self.matchThs, self.size_ranges, self.maxDets,
                                          self.matcher, self.matcher_kwargs, self.evaluate_img_separate)
        return res

    def get_ignore(self, gts):
        for gt in gts:
            gt['ignore'] = gt.get("iscrowd", 0)
            if self.use_ignore_attr:
                gt['ignore'] = gt["ignore"] or (gt.get('ignore', 0))  # changed by hui
        return np.array([gt['ignore'] for gt in gts], dtype=np.bool_)

    def summarize(self, res, gt_jd, print_func=None):
        if print_func is None:
            print_func = print
        try:
            from pycocotools.coco import COCO
            if isinstance(gt_jd, COCO):
                gt_jd = gt_jd.dataset
        except ModuleNotFoundError as e:
            pass
        assert isinstance(gt_jd, dict)

        all_aps = []
        all_ars = []
        for cls_i, (single_class_res, category) in enumerate(zip(res, gt_jd['categories'])):
            recalls = single_class_res['recall']
            precisions = single_class_res['precision']
            aps, ars = [], []
            for recall, precision in zip(recalls, precisions):
                ap = LocationEvaluator.get_AP_of_recall(recall, precision, recall_th=self.recThrs)
                aps.append(ap)
                ars.append(max(recall))
            all_aps.append(aps)
            all_ars.append(ars)
        all_aps = np.array(all_aps)
        all_ars = np.array(all_ars)

        if len(all_aps) > 0:
            mi = res[0]['match_th_idx']
            si = res[0]['size_range_idx']
            mdi = res[0]['maxDets_idx']
            if self.class_wise:
                self.print_class_wise(res, all_aps, all_ars, print_func)
            else:
                all_aps = all_aps.mean(axis=0)
                all_ars = all_ars.mean(axis=0)
                print(mi)
                for i, (ap, ar) in enumerate(zip(all_aps, all_ars)):
                    logs = "Location eval: (AP/AR) @[ dis={}\t| area={}\t| maxDets={}]\t= {}/{}".format(
                        self.matchThs[mi[i]], self.areaRngLbl[si[i]], self.maxDets[mdi[i]], '%.4f' % ap, '%.4f' % ar)
                    print_func(logs)

    def print_class_wise(self, res, all_aps, all_ars, print_func=None):
        for cls in range(len(res)):
            mi = res[cls]['match_th_idx']
            si = res[cls]['size_range_idx']
            mdi = res[cls]['maxDets_idx']
            for i, (ap, ar) in enumerate(zip(all_aps[cls], all_ars[cls])):
                logs = "({})Location eval: (AP/AR) @[ dis={}\t| area={}\t| maxDets={}]\t= {}/{}".format(
                    self.gt_jd['categories'][cls]['name'],
                    self.matchThs[mi[i]], self.areaRngLbl[si[i]], self.maxDets[mdi[i]], '%.4f' % ap, '%.4f' % ar)
                print_func(logs)

    @staticmethod
    def get_AP_of_recall(recall, precision, recall_th=None, DEBUG=False):
        assert len(recall) == len(precision), ""
        if recall_th is None:
            recall_th = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        elif isinstance(recall_th, int):
            recall_th = np.linspace(.0, 1.00, np.round((1.00 - .0) * recall_th) + 1, endpoint=True)
        inds = np.searchsorted(recall, recall_th, side='left')
        choose_precisions = [precision[pi] if pi < len(recall) else 0 for pi in inds]
        if DEBUG:
            print("choose_precisions", choose_precisions)
        return np.sum(choose_precisions) / len(recall_th)

    @staticmethod
    def add_center_from_bbox_if_no_point(det_jd):
        try:
            from pycocotools.coco import COCO
            if isinstance(det_jd, COCO):
                for idx, det in det_jd.anns.items():
                    if 'point' not in det:
                        x, y, w, h = det['bbox']
                        det['point'] = [x + (w - 1) / 2, y + (h - 1) / 2]
                        det_jd.anns[idx] = det
                return
        except ModuleNotFoundError as e:
            pass

        assert isinstance(det_jd, list)
        for det in det_jd:
            if 'point' not in det:
                x, y, w, h = det['bbox']
                det['point'] = [x + (w - 1) / 2, y + (h - 1) / 2]


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('det', help='det result file')
    parser.add_argument('gt', help='gt file')
    parser.add_argument('--matchThs', default=[0.5, 1.0, 2.0], nargs='+', type=float)
    parser.add_argument('--maxDets', default=[300], nargs='+', type=int)
    parser.add_argument('--class_wise', default=False, type=bool)
    parser.add_argument('--task', default=1, type=int)
    parser.add_argument('--given-recall', default=[0.9], nargs='+', type=float, help='arg for task==2')
    args = parser.parse_args()

    if isinstance(args.matchThs, float):
        args.matchThs = [args.matchThs]

    # ############################### 1. normal evaluation
    if args.task == 1:
        location_kwargs = dict(
            class_wise=args.class_wise,
            matcher_kwargs=dict(multi_match_not_false_alarm=False),
            location_param=dict(
                matchThs=args.matchThs,  # [0.5, 1.0, 2.0],
                recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                maxDets=args.maxDets,  # [300],
                # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                # maxDets=[1000],
            )
        )

        print(location_kwargs)
        # '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json'
        # exp//latest_result.json
        gt_file = args.gt  # '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json'
        det_file = args.det  # '/home/ubuntu/github/sparsercnn/outputs/locanet/visdroneperson_sparsercnn.res50.1000pro/' \
                  #  '640_stridein3_ADAMW_1x/inference/coco_instances_results.json'
        import json
        gt_jd = json.load(open(gt_file))
        det_jd = json.load(open(det_file))
        LocationEvaluator.add_center_from_bbox_if_no_point(det_jd)
        loc_evaluator = LocationEvaluator(**location_kwargs)
        res = loc_evaluator(det_jd, gt_jd)
        loc_evaluator.summarize(res, gt_jd)
    # ############################################# 2. find score with given recall
    elif args.task == 2:
        location_kwargs = dict(
            matcher_kwargs=dict(multi_match_not_false_alarm=False),
            location_param=dict(
                matchThs=args.matchThs,  # [0.5, 1.0, 2.0],
                recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                maxDets=args.maxDets,  # [300],
                # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                # maxDets=[1000],
                areaRng=[[1 ** 2, 1e5 ** 2]],
                areaRngLbl=['all'],
            )
        )
        LocationEvaluator.SAVE_RECALL_PRECISION_PATH = "/tmp/evaluation.npz"

        print(location_kwargs)
        # '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json'
        # exp//latest_result.json
        gt_file = args.gt  # '/home/ubuntu/dataset/visDrone/coco_fmt_annotations/VisDrone2018-DET-val-person.json'
        det_file = args.det  # '/home/ubuntu/github/sparsercnn/outputs/locanet/visdroneperson_sparsercnn.res50.1000pro/' \
                  #  '640_stridein3_ADAMW_1x/inference/coco_instances_results.json'
        import json
        gt_jd = json.load(open(gt_file))
        det_jd = json.load(open(det_file))
        LocationEvaluator.add_center_from_bbox_if_no_point(det_jd)
        loc_evaluator = LocationEvaluator(**location_kwargs)

        res = loc_evaluator(det_jd, gt_jd)
        loc_evaluator.summarize(res, gt_jd)

        #
        import matplotlib.pyplot as plt
        d = np.load(LocationEvaluator.SAVE_RECALL_PRECISION_PATH)
        dr = d['recall']
        for given_recall in args.given_recall:
            idx = np.arange(0, len(dr), 1)[dr >= given_recall][0]
            print('recall, precision, score:', dr[idx], d['precision'][idx], d['dets_score'][idx])

    # import sys
    # # ########## small test1 begin #########################################################################
    # S, D = [0.9, 0.89, 0.7], [(0.79, 0.7), (0.1, 0.2), (0.498, 0.498)]
    # S, D = np.array(S), np.array(D)
    # gts = np.array([(0.5, 0.5, 0.1, 0.2), (0.7, 0.7, 0.2, 0.1)])
    # gts_ignore = np.array([False, False])
    #
    # ignore_gts = gts[gts_ignore]
    # gts = gts[np.logical_not(gts_ignore)]
    #
    # # D = descend_sort_by_score(D)
    # idx = np.argsort(-S)
    # D = D[idx]
    # S = S[idx]
    #
    # matcher = PointMatcher(LOG=sys.stdout)
    # print('[test]: dis_th = 1, multi_match_not_false_alarm=False')
    # M, ID, det_scores = matcher(D, S, gts, ignore_gts, 1, False)
    # print("match_gt_id and is_ignore_det", M, D)
    #
    # matcher = PointMatcher(LOG=sys.stdout)
    # print('[test]: dis_th = 5, multi_match_not_false_alarm=False')
    # print("match_gt_id and is_ignore_det", matcher(D, S, gts, ignore_gts, 5, False))
    #
    # matcher = PointMatcher(LOG=sys.stdout)
    # print('[test]: dis_th = 5, multi_match_not_false_alarm=True')
    # print("match_gt_id and is_ignore_det", matcher(D, S, gts, ignore_gts, 5, True))
    # # ########## small test1 over ############################################################################
    #
    # # all full test2: apply point evaluation detection
    # root_dir = "/home/data/github/tiny_benchmark/tiny_benchmark/outputs/tiny_set/"
    # # maskrcnn_benchmark format output of bbox detection
    # det_file = root_dir + "FPN/baseline3_R101_cocov3_DA_t_s2.5x_a8/inference/" \
    #                       "tiny_set_corner_sw640_sh512_test_all_coco/bbox_merge_nms0.5.json"
    #
    # data_root_dir = "/home/data/github/TinyObject/Tiny/add_dataset/_final_dataset/"
    # gt_file = data_root_dir + "annotations/task/tiny_set_test_all.json"
    # import json
    #
    # # from pycocotools.coco import COCO
    # # gt_jd = COCO(gt_file)
    # # det_jd = gt_jd.loadRes(det_file)
    #
    # gt_jd = json.load(open(gt_file))
    # det_jd = json.load(open(det_file))
    # LocationEvaluator.add_center_from_bbox_if_no_point(det_jd)
    #
    # MAX_SIZE = 1e9
    # evaluator = LocationEvaluator(
    #     size_ranges=[(1, 20), (20, MAX_SIZE), (1, MAX_SIZE)],
    #     match_th_list=[0.5, 1.0, 2.0],
    #     multi_match_not_false_alarm=False
    # )
    #
    # res = evaluator(det_jd, gt_jd)
    # res = res[0]
    #
    # print(res.keys())
    #
    # # precision, recall = res['precision'][2], res['recall'][2]
    # import matplotlib.pyplot as plt
    #
    # for i in range(len(res['precision'])):
    #     plt.plot(res['recall'][i], res['precision'][i], label="{},{},{}".format(
    #         res['size_range'][i], res['match_th'][i], np.mean(res['precision'][i]).round(3)))
    # plt.legend()
    # plt.show()
    # # print(np.mean(precision))
