##aaa
./tools/dist_train.sh configs2/COCO/P2Seg/P2SegDistillBoxinst_r50_fpn_1x_coco.py 8 --work-dir=../TOV_mmdetection_cache/work_dirs/P2Seg/P2SegDistillBoxInst_mergeft --cfg-options model.roi_head.bbox_head.with_distill=2


work_dir=../TOV_mmdetection_cache/work_dirs/P2Seg/P2SegDistillBoxInst_new/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 ./tools/dist_train.sh configs2/COCO/P2Seg/P2SegDistillBoxinst_r50_fpn_1x_coco.py 8 --work-dir=${work_dir} --resume-from=/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/P2Seg/P2SegDistillBoxInst_new/epoch_12.pth --cfg-options evaluation.do_final_eval=True model.roi_head.bbox_head.with_distill=2
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/mask_rcnn_r50_fpn_2x_coco.py 8 --work-dir=${work_dir}'segmentation/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


###
./tools/dist_train.sh configs2/COCO/P2Seg/P2SegBoxinst_r50_fpn_1x_coco.py 8 --work-dir=../TOV_mmdetection_cache/work_dirs/P2Seg/P2SegBoxInst
