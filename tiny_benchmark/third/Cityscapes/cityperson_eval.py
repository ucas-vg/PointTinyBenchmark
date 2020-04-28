import json
import sys
import os

TOOL_DIR = os.path.abspath(__file__ + '/..') + '/evaluation/eval_script/'
sys.path.insert(0, TOOL_DIR)
import os
from coco import COCO
from eval_MR_multisetup import COCOeval, Params

"""
cityperson评价指标的特点：
1. 只评价cid=1的类别　eval_MR_multisetup.py: 346
2. 所有id从1开始计数,而不是0
2. det ignore, gt_ignore:
   如果评测时只评测高度大于50,那么det中高度小于50的就是det ignore,将不参与det-gt的匹配,相当于从检测结果里删除;
   而此时gt中除了本身就是gt ignore的,高度小于50的也会被设置为gt ignore,参与det-gt匹配,匹配上gt ignore的det从检测结果里移除,
        不参与tp/fp的统计;  eval_MR_multisetup.py:100
3. 命名十分不规范,使用A,K等极其简单的变量又缺少数字；代码大模块划分清晰，内部消磨块写成一团，大量使用for循环，缺乏矩阵编程功底
4. 修改 DetectionProject/notebook_mxnet/Cityscapes/evaluation/eval_script/eval_MR_multisetup.py:381,移除错误冗余的代码

5. 添加　按高度划分大小变为按size划分大小的选项
6. 添加　针对ignore gt的匹配, 在IOU后添加IOD的判断的选项
"""


def turn_bbox(src_file, dst_file):
    json_data = json.load(open(src_file))
    for data in json_data:
        data['image_id'] = data['image_id'] + 1
    json.dump(json_data, open(dst_file, 'w'))


def cityperson_eval(src_pth, annFile, CUT_WH=None,
                    ignore_uncertain=False, use_iod_for_ignore=False, catIds=[],
                    use_citypersons_standard=True, tiny_scale=1.0, iou_ths=None, setup_labels=None):
    if os.path.isdir(src_pth):
        resFile = src_pth + '/' + 'bbox.json'
    else:
        resFile = src_pth
    Params.CITYPERSON_STANDARD = use_citypersons_standard
    if use_citypersons_standard:
        kwargs = {}
        if CUT_WH is None: CUT_WH = (1, 1)
    else:
        kwargs = {'filter_type': 'size'}
        if CUT_WH is None: CUT_WH = (1, 1)
        Params.TINY_SCALE = tiny_scale
    Params.IOU_THS = iou_ths

    kwargs.update({'use_iod_for_ignore': use_iod_for_ignore, 'ignore_uncertain': ignore_uncertain})
    kwargs['given_catIds'] = len(catIds) > 0

    annType = 'bbox'      # specify type here
    print('Running demo for *%s* results.' % annType)

    # running evaluation
    print('CUT_WH:', CUT_WH)
    print('use_citypersons_standard:', use_citypersons_standard)
    print('tiny_scale:', tiny_scale)
    print(kwargs)
    res_file = open("results.txt", "w")
    Params.CUT_WH = CUT_WH
    setupLbl = Params().SetupLbl
    for id_setup in range(len(setupLbl)):
        if (setup_labels is None) or (setupLbl[id_setup] in setup_labels):
            cocoGt = COCO(annFile)
            cocoDt = cocoGt.loadRes(resFile)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt,cocoDt,annType, **kwargs)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate(id_setup)
            cocoEval.accumulate()
            cocoEval.summarize(id_setup,res_file)

    res_file.close()


if __name__ == '__main__':
    # for exp_name in ['FPN_RCNN_OHEM2', 'FPN_RPN_OHEM2', 'FPN_topk', 'FPN_topk_random', 'FPN'][:2]:
    #     cityperson_eval('../../outputs/perdestrian/{}/inference/cityscapes_perdestrian_val_cocostyle'.format(exp_name))
    # cityperson_eval('/home/data/github/maskrcnn-benchmark/outputs/perdestrian/FPN/inference/cityscapes_perdestrian_val_cocostyle')
    # '../../outputs/tiny/retiannet_all_rgb_cut/base/inference/tiny_all_rgb_cut_test_coco/'
    # test_cut = True
    # for exp_name in ['coco_pretrain_base4d2', 'coco_pretrain_v2_DA_2x', 'coco_pretrain_modify1', 'coco_pretrain_base4d2', 'base_4'][:1]:  # 'base', 'base_3', 'base_2',
    #     cityperson_eval('../../outputs/tiny/FPN_all_rgb_cut/{}/inference/tiny_all_rgb{}{}_test_coco'.format(
    #         exp_name, '_cut' if test_cut else '', '_modify1' if '_modify1' in exp_name else ''),
    #         '/home/hui/dataset/sanya/cocostyle_release/all/rgb/{}annotations/tiny_all_rgb{}_test_coco.json'
    #             .format('cut_' if test_cut else '', '_cut' if test_cut else ''),
    #         CUT_WH=(4, 2) if test_cut else (1, 1)
    #     )

    # cityperson_eval(
    #     '../../outputs/tiny_set/FPN/baseline1/inference/tiny_set_corner_sw640_sh512_test_all_coco/bbox_merge_nms0.5.json',
    #     '/home/hui/dataset/tiny_set/annotations/task/tiny_set_test_all.json', CUT_WH=(1, 1),
    #     ignore_uncertain=True, use_iod_for_ignore=True, catIds=[],
    #     use_citypersons_standard=False)

    cityperson_eval(
        # '../../outputs/cityperson/faster/base/inference/cityperson_pedestrian_val_coco/bbox.json',
        '/home/hui/github/cur_code/outputs/cityperson_FPN_baseline1.json',
        '/home/hui/dataset/cityscapes/perdestrian_annotations/citypersons_all_val.json', CUT_WH=(1, 1),
        ignore_uncertain=False, use_iod_for_ignore=False, catIds=[],
        use_citypersons_standard=True) #, tiny_scale=4.11886287119646)
