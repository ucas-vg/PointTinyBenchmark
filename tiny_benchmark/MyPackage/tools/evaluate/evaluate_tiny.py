from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import check_expected_results
from pycocotools.coco import COCO
from  maskrcnn_benchmark.data.datasets.evaluation.coco.cocoeval import COCOeval, Params
import os
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults
from third.Cityscapes.cityperson_eval import cityperson_eval


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
                ignore_uncertain=False, use_iod_for_ignore=False):
    return cityperson_eval(merged_det_file, merged_gt_file, CUT_WH=(1, 1),
                        ignore_uncertain=ignore_uncertain, use_iod_for_ignore=use_iod_for_ignore,
                        use_citypersons_standard=False)


# json_result_file = "/home/data/github/Face/FaceDetection-DSFD/outputs/Tinyperson640_DSFD_RES50/results/result.json"
json_result_file = '/home/hui/桌面/11.pkl.bbox.json'
corner_gt_file = "/home/hui/dataset/tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
merged_gt_file = '/home/hui/dataset/tiny_set/annotations/task/tiny_set_test_all.json'

print(os.path.split(json_result_file))
coco_gt, merged_det_file = merge_det_result(json_result_file, corner_gt_file, merged_gt_file, merge_nms_th=0.5)
print(merged_det_file)
# merged_det_file = '/home/data/github/Face/FaceDetection-DSFD/outputs/Tinyperson640_DSFD_RES50/results/result_merge_nms0.3.json'

results = evaluate_ap(merged_det_file, merged_gt_file, ignore_uncertain=True, use_iod_for_ignore=True,
                      eval_standard='tiny')
print(results)
evaluate_mr(merged_det_file, merged_gt_file, ignore_uncertain=True, use_iod_for_ignore=True)

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
