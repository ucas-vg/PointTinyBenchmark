### BOXINST
work_dir='../TOV_mmdetection_cache/work_dirs/P2Sweg/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNetBoxinst_r50_withoutfpn_1x_coco.py 2 \
--work-dir=${work_dir}  \
--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco14.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7