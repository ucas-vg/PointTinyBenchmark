import numpy as np
import datetime
import time
from collections import defaultdict
# from . import mask as maskUtils
import copy
import matplotlib.pyplot as plt
import scipy.io as sio


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm',
                 filter_type='height', use_iod_for_ignore=False,
                 ignore_uncertain=False, given_catIds=False):  # add by hui
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt              # ground truth COCO API
        self.cocoDt = cocoDt              # detections COCO API
        self.params = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            if not given_catIds:
                self.params.catIds = sorted(cocoGt.getCatIds())
        # #### add by hui ##########################################################################
        self.filter_type = filter_type
        assert filter_type in ['height', 'size'], "filter type must be 'height' or 'size'"
        self.use_iod_for_ignore = use_iod_for_ignore
        self.ignore_uncertain = ignore_uncertain
        # ##########################################################################################

    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            # ########################################################### change by hui ###############################
            import math
            if 'vis_ratio' not in gt:
                if 'vis' not in gt:
                    gt['vis_ratio'] = 1 - 1e-12
                else:
                    gt['vis_ratio'] = (gt['vis'][-1] / gt['bbox'][-1]) * (gt['vis'][-2] / gt['bbox'][-2])
            if 'size' not in gt:
                gt['size'] = math.sqrt(gt['area'])
            if 'height' not in gt:
                gt['height'] = gt['bbox'][-1]
            if self.filter_type == 'size':
                gt['ignore'] = 1 if (gt['size'] < self.params.HtRng[id_setup][0] or gt['size'] > self.params.HtRng[id_setup][1]) or \
                   (gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) else gt['ignore']
            else:
                gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or gt['height'] > self.params.HtRng[id_setup][1]) or \
                   (gt['vis_ratio'] < self.params.VisRng[id_setup][0] or gt['vis_ratio'] > self.params.VisRng[id_setup][1]) else gt['ignore']

            if self.ignore_uncertain and 'uncertain' in gt and gt['uncertain']:
                gt['ignore'] = 1
            ##########################################################

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval = {}                  # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
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

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId)\
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        VisRng = self.params.VisRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')


        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d,g,iscrowd)
        return ious

    def iou(self, dts, gts, pyiscrowd):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2,gx2)-max(dx1,gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2,gy2)-max(dy1,gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t)/unionarea
        return ious

    # ###### add by hui
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

    def IOD_by_IOU(self, dets, ignore_gts, ignore_gts_area, ious):
        if ignore_gts_area is None:
            ignore_gts_area = ignore_gts[:, 2] * dets[:, 3]
        dets_area = dets[:, 2] * dets[:, 3]
        tile_dets_area = np.tile(dets_area.reshape((-1, 1)), (1, len(ignore_gts_area)))
        tile_gts_area = np.tile(ignore_gts_area.reshape((1, -1)), (len(dets_area), 1))
        iods = ious / (1 + ious) * (1 + tile_gts_area / tile_dets_area)
        return iods
    ########

    def evaluateImg(self, imgId, catId, hRng, vRng, maxDet):
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
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]

        """
            changed code here.
        """
        # exclude dt out of height range ----0
        #d tind = np.array([int(d['id'] - dt[0]['id']) for d in dt])
        _dt, dtind = [], []
        for idx, d in enumerate(dt):
            # ################################ change by hui ###################################
            if self.filter_type == 'size':
                dsize = np.sqrt(d['bbox'][2] * d['bbox'][3])
                if dsize >= hRng[0] / self.params.expFilter and dsize < hRng[1] * self.params.expFilter:
                    _dt.append(d)
                    dtind.append(idx)
            else:
                if d['height'] >= hRng[0] / self.params.expFilter and d['height'] < hRng[1] * self.params.expFilter:
                    _dt.append(d)
                    dtind.append(idx)
            ###################################
        dt = _dt

        # load computed ious
        if len(dtind) > 0:
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]
        else:
            ious = []

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        ignore_gts = np.array([g['bbox'] for g in gt if g['_ignore']])
        ignore_gts_idx = np.array([i for i, g in enumerate(gt) if g['_ignore']])
        # #### ad by G ##############
        if len(ignore_gts_idx) > 0 and len(dt) > 0:
            ignore_gts_area = np.array([g['area'] for g in gt if g['_ignore']])  # use area
            ignore_ious = (ious.T[ignore_gts_idx]).T
        # #############################
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    bstOa = iou
                    bstg = -2
                    bstm = -2
                    for gind, g in enumerate(gt):
                        m = gtm[tind, gind]
                        # if this gt already matched, and not a crowd, continue
                        if m > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if bstm != -2 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < bstOa:
                            continue
                        # if match successful and best so far, store appropriately
                        bstOa = ious[dind,gind]
                        bstg = gind
                        if gtIg[gind] == 0:
                            bstm = 1
                        else:
                            bstm = -1

                    # if match made store id of match for both dt and gt
                    if bstg == -2:
                        # #### ad by hui ##############
                        if self.use_iod_for_ignore and len(ignore_gts) > 0:
                            # iods = self.IOD(np.array([d['bbox']]), ignore_gts)[0]
                            iods = self.IOD_by_IOU(np.array([d['bbox']]), None, ignore_gts_area,
                                                   ignore_ious[dind:dind+1, :])[0]
                            idx = np.argmax(iods)
                            if iods[idx] > 0.5:
                                # print('inside')
                                bstg = ignore_gts_idx[idx]
                                bstm = -1
                            else: continue
                        else: continue
                        ######################
                    dtIg[tind,dind] = gtIg[bstg]
                    dtm[tind,dind] = gt[bstg]['id']
                    if bstm == 1:
                        gtm[tind,bstg] = d['id']

        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'hRng':         hRng,
                'vRng':         vRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))  # -1 for the precision of absent categories

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)

        # retrieve E at each category, area range, and max number of detections
        add_gts_count = []  # add by hui
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')
                score_inds = inds.copy()
                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                add_gts_count.append(npig)  # add by hui
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                """ changed by hui"""
                # inds = np.where(dtIg==0)[1]
                # tps = tps[:,inds]
                # fps = fps[:,inds]
                """" """

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp)/I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

# #################################  add by hui #####################################################################
#                     import sys
#                     idx = np.array(range(fppi.shape[0]))[fppi >= 0.1]
#                     if len(idx) == 0:
#                         fppi01_score = dtScores[score_inds[-1]]
#                     else:
#                         idx = idx[0]
#                         if idx == 0:
#                             fppi01_score = dtScores[score_inds[0]] - 1e-12
#                         if fppi[idx] == 0.1:
#                             fppi01_score = dtScores[score_inds[idx]]
#                         else:
#                             fppi01_score = (dtScores[score_inds[idx-1]] + dtScores[score_inds[idx]]) / 2
#                         print(dtScores[score_inds[idx - 1]], dtScores[score_inds[idx]], file=sys.stderr)
#                         print(fppi[idx - 1], fppi[idx], file=sys.stderr)
#                     print("fppi == 0.1, score ==", fppi01_score, file=sys.stderr)
#######################################################################################################

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except:
                        pass
                    ys[t,:,k,m] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP':   ys,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        print('number of gt boxes: {}'.format(add_gts_count))

    def summarize(self,id_setup, res_file):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1-s[:,:,:,mind]

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)

            print(mrs.shape)#, mean_s >= 0.1)
            print(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
            res_file.write(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
            res_file.write('\n')
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        # _summarize(iouThr=.5,maxDets=1000)
        for iouThr in self.params.iouThrs:
            _summarize(iouThr=iouThr,maxDets=1000)

    def __str__(self):
        self.summarize()


# ################################ add by hui #############################################################
def get_fppi(min, max, n, EPS=1e-10):
    import numpy as np
    fppi_range = (min, max, n)
    log_step = np.log(fppi_range[1]/fppi_range[0])/(fppi_range[2]-1)
    fppis = np.exp(np.arange(np.log(fppi_range[0]), np.log(fppi_range[1])+EPS, log_step))
    assert len(fppis) == n, 'set EPS smaller please'
    return fppis
############################################################################################################


class Params:
    '''
    Params for coco evaluation api
    '''
    CUT_WH = (1, 1)  # add by hui
    CITYPERSON_STANDARD = False
    TINY_SCALE = 1
    IOU_THS = None
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []

        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        # ##########################3 add by hui /8 for 4*2 cut_image make a merged image #######################
        # np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])
        # self.fppiThrs = get_fppi(0.01, 8., 9) / Params.CUT_WH[0] / Params.CUT_WH[1]
        # self.fppiThrs = get_fppi(0.01, 1., 9) * 8/ Params.CUT_WH[0] / Params.CUT_WH[1] #
        self.fppiThrs = get_fppi(0.01, 1., 9) / Params.CUT_WH[0] / Params.CUT_WH[1] #
        #########################################################################################################
        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        if Params.CITYPERSON_STANDARD:
            self.iouThrs = np.array([0.5, 0.75])  if Params.IOU_THS is None else np.array(Params.IOU_THS) # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

            self.HtRng = [[50, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2], [20, 50]]
            self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2], [0.65, 1e5 ** 2]]
            self.SetupLbl = ['Reasonable', 'Reasonable_small','Reasonable_occ=heavy', 'All', 'small']
        else:
            self.iouThrs = np.array([0.25, 0.5, 0.75]) if Params.IOU_THS is None else np.array(Params.IOU_THS)

            s = Params.TINY_SCALE
            self.HtRng = [[2*s, 8*s], [8*s, 12*s], [12*s, 20*s], [2*s, 20*s], [20*s, 32*s], [-1, 1e5**2]]
            # self.HtRng = [[-1, 8], [8, 12], [12, 20], [-1, 20], [20, 32], [-1, 1e5**2]]
            # self.HtRng = [[40, 1e5 ** 2], [40, 1e5**2], [40,100], [-1, 40], [-1, 1e5**2]]
            self.VisRng = [[0.65, 1e5 ** 2] for _ in range(len(self.HtRng))]
            self.SetupLbl = ['tiny1', 'tiny2', 'tiny3', 'tiny', 'small', 'All']

            # self.HtRng = [[2*s, 20*s],]
            # # self.HtRng = [[-1, 8], [8, 12], [12, 20], [-1, 20], [20, 32], [-1, 1e5**2]]
            # # self.HtRng = [[40, 1e5 ** 2], [40, 1e5**2], [40,100], [-1, 40], [-1, 1e5**2]]
            # self.VisRng = [[0.65, 1e5 ** 2] for _ in range(len(self.HtRng))]
            # self.SetupLbl = ['tiny', ]

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

