from pycocotools.coco import COCO
import os
import argparse
import sys
import shutil
from huicv.deps.Cityscapes.cityperson_eval import cityperson_eval
from collections import OrderedDict
from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval


"""
require:
    conda install scipy, tqdm 
    
dependency install:
    cd huicv/deps/mini_maskrcnn_benchmark/ && python setup.py build develop

run example:
    python huicv/evaluation/evaluate_tiny.py --res exp/latest_result.json \
        --gt data/tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json \
        --merge-gt data/tiny_set/mini_annotations/tiny_set_test_all.json --detail
"""


class Log(object):
    log_map = {}

    def __init__(self, name):
        self.name = name

    def error(self, msg):
        print(f"[ERROR({self.name})]:", msg)

    def info(self, msg):
        print(f"[INFO({self.name})]:", msg)

    @staticmethod
    def getLogger(name):
        if name not in Log.log_map:
            Log.log_map[name] = Log(name)
        return Log.log_map[name]

# copy from maskrcnn_benchmark start ###########################################################
class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = Log.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
# copy from maskrcnn_benchmark end ###########################################################


def merge_det_result(json_result_file, corner_gt_file, merged_gt_file, merge_nms_th=1.0):
    from huicv.corner_dataset.split_and_merge_image import COCOMergeResult
    print('merge result from sub image', json_result_file, merged_gt_file)
    if merge_nms_th >= 1.0 - 1e-6:
        use_nms = False
    else:
        use_nms = True
    _, merged_json_result_file = COCOMergeResult(use_nms=use_nms, nms_th=merge_nms_th)(
        corner_gt_file,
        json_result_file,
        os.path.split(json_result_file)[0],  # dir
        merged_gt_file
    )
    coco_gt = COCO(merged_gt_file)
    return coco_gt, merged_json_result_file


def evaluate_ap(json_result_file, gt_file,
                iou_types=("bbox",), expected_results=(), expected_results_sigma_tol=4,
                iou_type="bbox", ignore_uncertain=False, use_iod_for_ignore=False, eval_standard='tiny'):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(str(json_result_file))

    # tiny evaluation
    cocofmt_kwargs=dict(
        ignore_uncertain=ignore_uncertain,
        use_ignore_attr=True,
        use_iod_for_ignore=use_iod_for_ignore,
        iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard=eval_standard,  # or 'coco'
            iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
            maxDets=[200],              # set this same as set evaluation.proposal_nums
        )
    )

    cocoEval = COCOExpandEval(coco_gt, coco_dt, iou_type, **cocofmt_kwargs)
    print(cocoEval.params.__dict__)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # coco_dt = coco_gt.loadRes(coco_results)
    # Params.EVAL_STRANDARD = eval_standard
    # coco_eval = COCOeval(coco_gt, coco_dt, iou_type, ignore_uncertain, use_iod_for_ignore)
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    #
    results = COCOResults(*iou_types)
    results.update(cocoEval)

    check_expected_results(results, expected_results, expected_results_sigma_tol)
    return results


# only support merged det_file and merged_gt_file
def evaluate_mr(merged_det_file, merged_gt_file,
                ignore_uncertain=False, use_iod_for_ignore=False, iou_ths=None, setup_labels=None):
    return cityperson_eval(merged_det_file, merged_gt_file, CUT_WH=(1, 1),
                           ignore_uncertain=ignore_uncertain, use_iod_for_ignore=use_iod_for_ignore,
                           use_citypersons_standard=False, iou_ths=iou_ths, setup_labels=setup_labels)


class RedirectStdOut(object):
    def __init__(self, file_name='/tmp/evaluate_tiny.log'):
        self.file_name = file_name
        self.stdout = sys.stdout

    def start(self):
        sys.stdout = open(self.file_name, 'w')

    def finish(self):
        filep = sys.stdout
        sys.stdout = self.stdout
        filep.close()


def rm_file(file_name):
    if os.path.exists(file_name):
        if os.path.isdir(file_name):
            shutil.rmtree(file_name)
        else:
            os.remove(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate AP and MR for Tiny Benchmark.')
    parser.add_argument('--res', dest='res', required=True, help='result file of detetor.')
    parser.add_argument('--gt', dest='gt', required=True, help='ground-truth file.')
    parser.add_argument('--merge-gt', dest='merge_gt', help='merged ground truth file.', default='')
    parser.add_argument('--metric', dest='metric', help='merged ground truth file.', default='all')
    parser.add_argument('--score-file', dest='score_file', help='output score file.', default='')
    parser.add_argument('--tmp-log', dest='tmp_log', help='temporal log file.', default='')
    parser.add_argument('--mr-ious', dest='mr_ious', help='iou th settings while evaluating MR.',
                        default='0.25,0.5,0.75')
    parser.add_argument('--mr-sizes', dest='mr_sizes', help='size settings while evaluating MR.',
                        default='tiny1,tiny2,tiny3,tiny,small,All')
    parser.add_argument('--detail', dest='detail', help='output detail info in result file', action='store_true')
    #     json_result_file = '/home/hui/桌面/11.pkl.bbox.json'
    #     corner_gt_file = "/home/hui/dataset/tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    #     merged_gt_file = '/home/hui/dataset/tiny_set/annotations/task/tiny_set_test_all.json'

    args = parser.parse_args()
    if len(args.tmp_log) == 0:
        args.tmp_log = os.path.join(os.path.dirname(args.res), 'tmp.log')
    if len(args.score_file) == 0:
        args.score_file = os.path.join(os.path.dirname(args.res), 'scores.txt')

    # merge res if needed.
    if len(args.merge_gt) > 0:
        _, det_file = merge_det_result(args.res, args.gt, args.merge_gt, merge_nms_th=0.5)
        print(det_file)
        gt_file = args.merge_gt
    else:
        det_file, gt_file = args.res, args.gt

    # evaluate and redirect output to tmp file
    rstdout = RedirectStdOut(args.tmp_log)
    rstdout.start()

    metric = args.metric.lower()
    if metric == 'ap' or metric == 'all':
        results = evaluate_ap(det_file, gt_file, ignore_uncertain=True, use_iod_for_ignore=True, eval_standard='tiny')
    if metric == 'mr' or metric == 'all':
        iou_ths = [float(x) for x in args.mr_ious.split(',') if len(x.strip()) > 0]
        setup_labels = [x.strip() for x in args.mr_sizes.split(',') if len(x.strip()) > 0]
        evaluate_mr(det_file, gt_file, ignore_uncertain=True, use_iod_for_ignore=True, iou_ths=iou_ths,
                    setup_labels=setup_labels)

    rstdout.finish()
    rm_file(os.path.join(os.path.dirname(__file__), 'results.txt'))

    if not args.detail:
        from MyPackage.visulize.plot_train_log import parse_log, replace_key

        # parse log and write it to score.txt
        res = parse_log(args.tmp_log)
        res = replace_key(res)
        res = {k: v[0] for k, v in res.items() if len(k) < 15}
        with open(args.score_file, 'w') as f:
            for k, v in res.items():
                f.write("{}: {}\n".format(k, float(v)))
        rm_file(args.tmp_log)
    else:
        os.rename(args.tmp_log, args.score_file)

# # generate ignore
# jd = json.load(open("/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007.json"))
# jd['annotations'] = [ann for ann in jd['annotations'] if ann['ignore'] == 0]
# json.dump(jd, open('/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007_noignore.json', 'w'))
#
# jd = json.load(open("/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007.json"))
# for ann in jd['annotations']:
#     ann['ignore'] = 0
# json.dump(jd, open('/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007_useignore.json', 'w'))
#
# evaluate_ap("/home/hui/github/cur_code/outputs/pascal/gau/base_LD2.4/inference/voc_2007_test_cocostyle/bbox.json",
#             "/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007.json",
#             eval_standard='coco')
