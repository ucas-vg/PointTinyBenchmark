
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 \
--work-dir '../TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/detection/without_weight' \
--cfg-options data.train.ann_file="../TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/cooc_1200_pseudo_ann.json"
python exp/tools/killgpu.py 0-7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 \
--work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/detection/without_weight' \
--cfg-options data.train.ann_file="../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/coco_1200_latest_pseudo_ann.json"
python exp/tools/killgpu.py 0-7
