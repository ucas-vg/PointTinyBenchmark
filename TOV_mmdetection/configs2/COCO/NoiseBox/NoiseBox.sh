## Noise 0.4
work_dir='../TOV_mmdetection_cache/work_dirs/NoiseBox/COCO/noise_ann_with_pbr_2stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/NoiseBox/NoiseBox_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json'  \
#--resume-from='../TOV_mmdetection_cache/work_dirs/NoiseBox/COCO/noise_ann_with_pbr_2stage/epoch_4.pth'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


## Noise 0.2
noise_level=0.4 && work_dir='../TOV_mmdetection_cache/work_dirs/NoiseBox/COCO/noise_0.4/noise_0.4_ann_sr0.1_br2_with_pbr_2stage/'  && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/NoiseBox_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' \
data.train.box_noise_level=${noise_level} data.val.box_noise_level=${noise_level}
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json' \
#--resume-from='../TOV_mmdetection_cache/work_dirs/NoiseBox/COCO/noise_0.2/noise_0.2_ann_sr0.1_with_pbr_1stage/detection2/without_weight/epoch_11.pth'
python exp/tools/killgpu.py 0-7




#ENoiseBox+reg
noise_level=0.4 && cluster_mode='classification' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0_with_reg_${cluster_mode}/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco.py 8 \
--work-dir=${work_dir}  \
--cfg-options   model.roi_head.cluster_mode=${cluster_mode} model.roi_head.bbox_head.loss_bbox.loss_weight=1.0

##EnpiseBox
noise_level=0.4  && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBox_r50_fpn_1x_coco.py 8 \
--work-dir=${work_dir} --resume-from=${work_dir}'epoch_2.pth'
noise_level=0.4 &&  work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBox_r50_fpn_1x_coco_retrain.py 8 \
--work-dir=${work_dir}  --resume-from=${work_dir}'epoch_12.pth'  \
--cfg-options  evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

#ENoiseBox+reg+retrain
noise_level=0.4 && cluster_mode='upper' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0_with_reg_${cluster_mode}/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco.py 8 \
--work-dir=${work_dir} \
--cfg-options   model.roi_head.cluster_mode=${cluster_mode} model.roi_head.bbox_head.loss_bbox.loss_weight=1.0
python exp/tools/killgpu.py 0-7
noise_level=0.4 && cluster_mode='upper' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0_with_reg_${cluster_mode}/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_retrain.py 8 \
--work-dir=${work_dir}  --resume-from=${work_dir}'epoch_12.pth'  \
--cfg-options  evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


##EnpiseBox+retrain
noise_level=0.4  && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBox_r50_fpn_1x_coco.py 8 \
--work-dir=${work_dir}
python exp/tools/killgpu.py 0-7
noise_level=0.4 &&  work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_retain.py 8 \
--work-dir=${work_dir}  --resume-from=${work_dir}'epoch_12.pth'  \
--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' evaluation.do_final_eval=True
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


### 0.2
noise_level=0.2 && cluster_mode='handdistill0.25+distill0.25+weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.1_br2_with_pbr_2stage_det4.0${cluster_mode}/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_0.2.py 8 --work-dir=${work_dir} --cfg-options model.roi_head.bbox_head.with_distill=False data.train.ann_file=data/coco/noisy_pkl/instances_train2017_noise-r0.2.json
python exp/tools/killgpu.py 0-7

noise_level=0.2 && cluster_mode='handdistill0.25+distill0.25+weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.1_br2_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_retrain_0.2.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True --resume-from=${work_dir}'epoch_12.pth'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

noise_level=0.2 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10012 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_retrain_0.2.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True --resume-from=${work_dir}'epoch_12.pth'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


noise_level=0.4 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_r101' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10012 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r101_fpn_1x_coco_retrain.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True --resume-from=${work_dir}'epoch_12.pth'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

noise_level=0.4 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_r101' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10212 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r101_fpn_1x_coco_retrain.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True --resume-from=${work_dir}'epoch_12.pth'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_800_latest_result.json' ${work_dir}'coco_800_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10203 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

## 0.4
noise_level=0.4 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco_ms.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True

noise_level=0.4 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_r101' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r101_fpn_1x_coco_.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True
#data.workers_per_gpu=0

noise_level=0.4 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_r101' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/COCO/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/NoiseBox/ENoiseBoxPLUS_r101_fpn_1x_coco.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True --resume-from=${work_dir}'epoch_6.pth'
#data.workers_per_gpu=0


noise_level=0.1 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/VOC/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_voc_0.1.py 8 --work-dir=${work_dir} \
--cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True
#data.workers_per_gpu=0


noise_level=0.1 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/VOC/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ && CUDA_VISIBLE_DEVICES=6,7 PORT=18001 tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_voc_0.1.py 8 --work-dir=${work_dir} --cfg-options evaluation.save_result_file=${work_dir}'_800_latest_result.json' evaluation.do_final_eval=True model.roi_head.with_objectness=True

noise_level=0.1 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/VOC/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}/ && CUDA_VISIBLE_DEVICES=6,7 PORT=18004 tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_voc_${noise_level}.py 2 --work-dir=${work_dir}  evaluation.do_final_eval=True model.roi_head.with_objectness=True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10305 ./tools/dist_train.sh configs2/VOC/detection/faster_rcnn_r50_fpn_1x_VOC.py 2 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/Untitled_Folder/voc_800_latest_pseudo_ann_0.3.json

noise_level=0.3 && cluster_mode='_handdistill0.25+distill0.25+weight_objectness_plus_weight' && work_dir=../TOV_mmdetection_cache/work_dirs/ENoiseBox/VOC/SS/noise_${noise_level}/noise_${noise_level}_ann_sr0.2_br1_with_pbr_2stage_det4.0${cluster_mode}
