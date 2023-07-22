from collections import defaultdict
from pycocotools.cocoeval import Params, COCOeval
import numpy as np


class ExpandParam(Params):
    '''
    Params for coco evaluation api
    '''

    def setDetParams(self):
        eval_standard = self.evaluate_standard.lower()
        if eval_standard.startswith('tiny'):
            self.imgIds = []
            self.catIds = []
            # np.arange causes trouble.  the data point on arange is slightly larger than the true value
            if eval_standard == 'tiny':
                self.iouThrs = np.array([0.25, 0.5, 0.75])
            elif eval_standard == 'tiny_sanya17':
                self.iouThrs = np.array([0.3, 0.5, 0.75])
            else:
                raise ValueError(
                    "eval_standard is not right: {}, must be 'tiny' or 'tiny_sanya17'".format(eval_standard))
            self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
            self.maxDets = [200]
            self.areaRng = [[1 ** 2, 1e5 ** 2], [1 ** 2, 20 ** 2], [1 ** 2, 8 ** 2], [8 ** 2, 12 ** 2],
                            [12 ** 2, 20 ** 2], [20 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
            # s = 4.11886287119646
            # self.areaRng = np.array(self.areaRng) * (s ** 2)
            self.areaRngLbl = ['all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable']
            self.useCats = 1
        elif eval_standard == 'coco':  # COCO standard
            self.imgIds = []
            self.catIds = []
            # np.arange causes trouble.  the data point on arange is slightly larger than the true value
            self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
            self.maxDets = [1, 10, 100]
            self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
            self.areaRngLbl = ['all', 'small', 'medium', 'large']
            self.useCats = 1
            # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 10 ** 2], [10 ** 2, 32 ** 2],
            #                 [32 ** 2, 96 ** 2], [96 ** 2, 288 ** 2], [288 ** 2, 1e5 ** 2]]
            # self.areaRngLbl = ['all', 'out_small', 'in_small', 'medium', 'in_large', 'out_large']
            #
            # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 64 ** 2], [64 ** 2, 128 ** 2],
            #                 [128 ** 2, 256 ** 2], [256 ** 2, 1024 ** 2], [1024 ** 2, 1e5 ** 2]]
            # self.areaRngLbl = ['all', 'smallest', 'fpn1', 'fpn2', 'fpn3', 'fpn4+5', 'largest']
        else:
            print("use self define evaluate standard.")
            for key in ['imgIds', 'catIds', 'iouThrs', 'recThrs', 'maxDets', 'areaRng', 'areaRngLbl', 'useCats']:
                assert key in self.__dict__, "'{}' must be given in cocofmt_param " \
                                             "while 'evaluate_standard' is not 'coco' or 'tiny'".format(key)
            # raise ValueError('evaluate_standard not valid.')

    def __init__(self, iouType='segm', evaluate_standard='coco', **kwargs):   # add by hui
        self.evaluate_standard = evaluate_standard  # add by hui
        super(ExpandParam, self).__init__(iouType)
        # ##########################################
        for key, value in kwargs.items():
            assert key in ['imgIds', 'catIds', 'iouThrs', 'recThrs', 'maxDets', 'areaRng', 'areaRngLbl', 'useCats'],\
                "{} not args for Param".format(key)
            if key in ['iouThrs', 'recThrs']:
                value = np.array(value)
            self.__dict__[key] = value


class COCOExpandEval(COCOeval):
    """
    some modified:
    1. gt['ignore'], use_ignore_attr
        use_ignore_attr=False, same as COCOeval: if 'iscrowd' and 'ignore' all set in json file, only use 'iscrowd'
        use_ignore_attr=True: if 'iscrowd' and 'ignore' all set in json file, use ('iscrowd' | 'ignore')
    2. ignore_uncertain
        if 'uncertain' key set in json file, this flag control whether treat gt['ignore'] of 'uncertain' bbox as True
    3. use_iod_for_ignore
        whether use 'iod' evaluation standard while match with 'ignore' bbox
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm',
                 ignore_uncertain=False, use_ignore_attr=False,
                 use_iod_for_ignore=False, iod_th_of_iou_f="lambda iou: iou",
                 cocofmt_param={}):  # add by hui
        """
            iod_th_of_iou_f=lambda iou: iou, use same th of iou as th of iod
            iod_th_of_iou_f=lambda iou: (2*iou)/(1+iou), iou = I/(I+xD+xG), iod=I/(I+xD),
            we assume xD=xG, then iod=(2*iou)/(1+iou)
        """
        super(COCOExpandEval, self).__init__(cocoGt, cocoDt, iouType)
        self.use_ignore_attr = use_ignore_attr
        self.use_iod_for_ignore = use_iod_for_ignore
        self.ignore_uncertain = ignore_uncertain
        self.iod_th_of_iou_f = eval(iod_th_of_iou_f)
        self.params = ExpandParam(iouType=iouType, **cocofmt_param)  # parameters
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            if self.use_ignore_attr:
                gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
                gt['ignore'] = ('iscrowd' in gt and gt['iscrowd']) or gt['ignore']  # changed by hui
            else:
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

            # ########################################################### change by hui ###############################
            if self.ignore_uncertain and 'uncertain' in gt and gt['uncertain']:
                gt['ignore'] = 1
            # ########################################################### change by hui ###############################
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    # ###### add by G
    def IOD(self, dets, ignore_gts):
        def insect_boxes(box1, boxes):
            sx1, sy1, sx2, sy2 = box1[:4]
            tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            ix1 = np.where(tx1 > sx1, tx1, sx1)
            iy1 = np.where(ty1 > sy1, ty1, sy1)
            ix2 = np.where(tx2 < sx2, tx2, sx2)
            iy2 = np.where(ty2 < sy2, ty2, sy2)
            return np.array([ix1, iy1, ix2, iy2]).transpose((1, 0))

        def bbox_area(boxes):
            s = np.zeros(shape=(boxes.shape[0],), dtype=np.float32)
            tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            h = (tx2 - tx1)
            w = (ty2 - ty1)
            valid = np.all(np.array([h > 0, w > 0]), axis=0)
            s[valid] = (h * w)[valid]
            return s

        def bbox_iod(dets, gts, eps=1e-12):
            iods = np.zeros(shape=(dets.shape[0], gts.shape[0]), dtype=np.float32)
            dareas = bbox_area(dets)
            for i, (darea, det) in enumerate(zip(dareas, dets)):
                idet = insect_boxes(det, gts)
                iarea = bbox_area(idet)
                iods[i, :] = iarea / (darea + eps)
            return iods

        def xywh2xyxy(boxes):
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            return boxes

        from copy import deepcopy
        return bbox_iod(xywh2xyxy(deepcopy(dets)), xywh2xyxy(deepcopy(ignore_gts)))

    # add by hui
    def IOD_by_IOU(self, dets, ignore_gts, ignore_gts_area, ious):
        if ignore_gts_area is None:
            ignore_gts_area = ignore_gts[:, 2] * dets[:, 3]
        dets_area = dets[:, 2] * dets[:, 3]
        tile_dets_area = np.tile(dets_area.reshape((-1, 1)), (1, len(ignore_gts_area)))
        tile_gts_area = np.tile(ignore_gts_area.reshape((1, -1)), (len(dets_area), 1))
        iods = ious / (1 + ious) * (1 + tile_gts_area / tile_dets_area)
        return iods

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        # #### ad by hui ##############
        ignore_gts = np.array([g['bbox'] for g in gt if g['_ignore']])
        ignore_gts_idx = np.array([i for i, g in enumerate(gt) if g['_ignore']])
        if len(ignore_gts_idx) > 0 and len(dt) > 0:
            ignore_gts_area = np.array([g['area'] for g in gt if g['_ignore']])  # use area
            ignore_ious = (ious.T[ignore_gts_idx]).T
        ######################
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        # #### ad by hui ##############
                        if self.use_iod_for_ignore and len(ignore_gts) > 0:
                            # time from 156.88s -> 79. s
                            iods = self.IOD_by_IOU(np.array([d['bbox']]), None, ignore_gts_area,
                                                   ignore_ious[dind:dind + 1, :])[0]
                            # iods =self.IOD(np.array([d['bbox']]), ignore_gts)[0]
                            idx = np.argmax(iods)
                            if iods[idx] >= self.iod_th_of_iou_f(iou):
                                # print('inside')
                                m = ignore_gts_idx[idx]

                                dtIg[tind, dind] = gtIg[m]
                                dtm[tind, dind] = gt[m]['id']
                                gtm[tind, m] = d['id']
                            else:
                                continue
                        else:
                            continue
                        ######################
                        ###continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def summarize(self, print_func=print):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def float_equal(a, b):
            return np.abs(a-b) < 1e-6

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'  # change by hui {:0.3f} to {:0.4f}
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(float_equal(iouThr, p.iouThrs))[0] # changed by hui
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(float_equal(iouThr, p.iouThrs))[0] # t = np.where(iouThr == p.iouThrs)[0] changed by hui
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print_func(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        # ###################################### changed by hui ###########################################
        def _summarizeDets_tiny():
            stats = []
            for isap in [1, 0]:
                for iouTh in self.params.iouThrs:
                    for areaRng in self.params.areaRngLbl:
                        stats.append(_summarize(isap, iouThr=iouTh, areaRng=areaRng, maxDets=self.params.maxDets[-1]))
            return np.array(stats)
        def _summarizeDets():
            # stats = np.zeros((12,))
            # stats[0] = _summarize(1)
            # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            n = len(self.params.areaRngLbl)-1
            stats = []
            stats.extend([_summarize(1),
                          _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.6, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.7, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2]),
                          _summarize(1, iouThr=.9, maxDets=self.params.maxDets[2])])
            for i in range(n):
                # stats.append(_summarize(1, iouThr=0.5, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
                stats.append(_summarize(1, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
            stats.extend([_summarize(0, maxDets=self.params.maxDets[0]),
                          _summarize(0, maxDets=self.params.maxDets[1]),
                          _summarize(0, maxDets=self.params.maxDets[2])])
            for i in range(n):
                stats.append(_summarize(0, areaRng=self.params.areaRngLbl[i+1], maxDets=self.params.maxDets[2]))
            return stats
        # #####################################################################################
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            if self.params.evaluate_standard.startswith('tiny'): summarize = _summarizeDets_tiny
            else: summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        import time, copy
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))


if __name__ == '__main__':
    from pycocotools.coco import COCO
    gt_file = 'data/tiny_set/mini_annotations/tiny_set_test_all.json'
    res_file = 'exp/latest_result.json'
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(res_file)

    # origin coco evaluation
    # cocofmt_kwargs = {}

    # tiny evaluation
    cocofmt_kwargs=dict(
        ignore_uncertain=True,
        use_ignore_attr=True,
        use_iod_for_ignore=True,
        iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard='tiny',  # or 'coco'
            iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
            maxDets=[200],              # set this same as set evaluation.proposal_nums
        )
    )

    cocoEval = COCOExpandEval(cocoGt, cocoDt, 'bbox', **cocofmt_kwargs)
    print(cocoEval.params.__dict__)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
