from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import check_expected_results
from pycocotools.coco import COCO
from  maskrcnn_benchmark.data.datasets.evaluation.coco.cocoeval import COCOeval, Params
import os
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults
from third.Cityscapes.cityperson_eval import cityperson_eval
import argparse
import sys
import shutil

def merge_det_result(json_result_file, corner_gt_file, merged_gt_file, merge_nms_th=1.0):
    from MyPackage.tools.pub.split_and_merge_image import COCOMergeResult
    print('merge result from sub image', json_result_file, merged_gt_file)
    if merge_nms_th >= 1.0 - 1e-6:
        use_nms = False
    else:
        use_nms = True
    _, merged_json_result_file = COCOMergeResult(use_nms=use_nms, nms_th=merge_nms_th)(
        corner_gt_file,
        json_result_file,
        os.path.split(json_result_file)[0]  # dir
    )
    coco_gt = COCO(merged_gt_file)
    return coco_gt, merged_json_result_file


def evaluate_ap(json_result_file, gt_file,
             iou_types=("bbox",), expected_results=(), expected_results_sigma_tol=4,
             iou_type="bbox", ignore_uncertain=False, use_iod_for_ignore=False, eval_standard='tiny'):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(str(json_result_file))

    # coco_dt = coco_gt.loadRes(coco_results)
    Params.EVAL_STRANDARD = eval_standard
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type, ignore_uncertain, use_iod_for_ignore)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    #
    results = COCOResults(*iou_types)
    results.update(coco_eval)

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
    parser.add_argument('--mr-ious', dest='mr_ious', help='iou th settings while evaluating MR.', default='0.25,0.5,0.75')
    parser.add_argument('--mr-sizes', dest='mr_sizes', help='size settings while evaluating MR.', default='tiny1,tiny2,tiny3,tiny,small,All')
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
    if metric == 'ap' or metric =='all':
        results = evaluate_ap(det_file, gt_file, ignore_uncertain=True, use_iod_for_ignore=True, eval_standard='tiny')
    if metric == 'mr' or metric == 'all':
        iou_ths = [float(x) for x in args.mr_ious.split(',') if len(x.strip()) > 0]
        setup_labels = [x.strip() for x in args.mr_sizes.split(',') if len(x.strip()) > 0]
        evaluate_mr(det_file, gt_file, ignore_uncertain=True, use_iod_for_ignore=True, iou_ths=iou_ths, setup_labels=setup_labels)
    
    rstdout.finish()
    rm_file(os.path.join(os.path.dirname(__file__), 'results.txt'))

    if not args.detail:
        from MyPackage.visulize.plot_train_log import parse_log, replace_key
    
        # parse log and write it to score.txt
        res = parse_log(args.tmp_log)
        res = replace_key(res)
        res = {k:v[0] for k, v in res.items() if len(k) < 15}
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
