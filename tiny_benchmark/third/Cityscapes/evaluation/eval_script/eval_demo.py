from coco import COCO
from eval_MR_multisetup import COCOeval

annType = 'bbox'      #specify type here
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
annFile = '../val_gt.json'
# initialize COCO detections api
# resFile = '../val_dt.json'
# resFile = './coco_citypersons_val_citypersons_1gpu_e2e_faster_rcnn_R-50-FPN_v1_60-69999_dt_sort.json'
resFile = '/home/data/github/maskrcnn-benchmark/outputs/perdestrian/FPN/inference/cityscapes_perdestrian_val_cocostyle/bbox.json'

## running evaluation
res_file = open("results.txt", "w")
for id_setup in range(0,4):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate(id_setup)
    cocoEval.accumulate()
    cocoEval.summarize(id_setup,res_file)

res_file.close()
