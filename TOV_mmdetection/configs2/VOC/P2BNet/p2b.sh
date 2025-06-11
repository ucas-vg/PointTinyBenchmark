work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_1/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.02
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_2/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.01
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_3/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.005
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_4_top4/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.top_k=4 model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.002
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py /home/pfchen/disk1/cpf/P2BNet/TOV_mmdetection/data/PascalVOC_coco/voc07_trainval.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/VOC/detection/faster_rcnn_r50_fpn_1x_VOC.py 2 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'epoch_12.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_5/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.001
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/VOC_07/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_5/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/VOC/P2BNet/P2BNet_r50_fpn_1x_VOC_07_ms.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.0005
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7