./tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2

./tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2 --ops

CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2 \
--work-dir '/home/pfchen/yxh/TOV_mmdetection_cache/work_dir/TinyCOCO/P2B/softmax_with_fpn_8_64' \
--cfg-options model.train_cfg.rpn_proposal.base_scales=[8,16,32,64]
TOV_mmdetection

TOV_mmdetectionTOV_mmdetection

##test test_scale=[480,576,688,864,1000,1200]
export test_scale=480
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7TOV_mmdetection

export test_scale=688
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=864
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=1200
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7



## p2b + det atten
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten' \
--cfg-options model.roi_head.with_atten=True 
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/_1200_latest_result.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/coco_1200_latest_pseudo_ann.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/detection/without_weight' --cfg-options data.train.ann_file="../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/coco_1200_latest_pseudo_ann.json" python exp/tools/killgpu.py 0-7

## p2b + det  1 stage
export work_dir = '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=work_dir \
--cfg-options model.roi_head.with_atten=False  model.roi_head.num_stages=1 model.roi_head.bbox_head.num_stages=1  evaluation.save_result_file=work_dir + '_1200' + '_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/_1200_latest_result.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/coco_1200_latest_pseudo_ann.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/detection/without_weight' --cfg-options data.train.ann_file="../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/coco_1200_latest_pseudo_ann.json" python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo without neg
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_withoutneg/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.train_cfg.fine_proposal.gen_num_neg=0 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudoTOV_mmdetection_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  vgg16TOV_mmdetection
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/vgg16_ms_cas_sharefc_softmax_sigmoidneg_milmil/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_vgg16_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


## p2b + det  r101 without pseudo
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_r101_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r101_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False   model.roi_head.bbox_head.with_loss_pseudo=False  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + 3stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=3 model.roi_head.bbox_head.num_stages=3  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + 4stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_4stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=4 model.roi_head.bbox_head.num_stages=4  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7

##test
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/test/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.0002 load_from='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_withoutneg/epoch_12.pth'

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/test/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=3 model.roi_head.bbox_head.num_stages=3  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.0002 load_from='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/epoch_12.pth'

## p2b + det  without pseudo +topk k=3 7 10
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_top10/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.roi_head.top_k=10 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo without neg + mil-1 in stage2
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_mil1stage2/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7



###########################################################################################################
./tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2

./tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2 --ops

CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyCOCO/P2BNet/P2BNet_r50_fpn_1x_coco.py 2 \
--work-dir '/home/pfchen/yxh/TOV_mmdetection_cache/work_dir/TinyCOCO/P2B/softmax_with_fpn_8_64' \
--cfg-options model.train_cfg.rpn_proposal.base_scales=[8,16,32,64]




##test test_scale=[480,576,688,864,1000,1200]
export test_scale=480
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=576
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=688
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=864
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7

export test_scale=1200
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir='/home/ubuntu/cpf/P2BNet/TOV_mmdetection_cache/work_dirs/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_reshape/test_result/'
--cfg-options test_scale=test_scale test_pipeline.image_scale=(2000,test_scale) evaluation.save_result_file=work_dir+'_'+str(test_scale)+'_latest_result.json'
python exp/tools/killgpu.py 0-7



## p2b + det atten
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten' \
--cfg-options model.roi_head.with_atten=True 
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/_1200_latest_result.json ../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/coco_1200_latest_pseudo_ann.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/detection/without_weight' --cfg-options data.train.ann_file="../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_reshape_atten/coco_1200_latest_pseudo_ann.json" python exp/tools/killgpu.py 0-7

## p2b + det  1 stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/'&& 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.num_stages=1 model.roi_head.bbox_head.num_stages=1  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'epoch_7.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  1 stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_mil/'&& 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.num_stages=1 model.roi_head.bbox_head.num_stages=1  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'epoch_7.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo without neg
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_withoutneg/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.train_cfg.fine_proposal.gen_num_neg=0 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  vgg16
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/vgg16_ms_cas_sharefc_softmax_sigmoidneg_milmil/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_vgg16_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py /home/ubuntu/mnt/dataset/MSCOCO2017/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann.json'
python exp/tools/killgpu.py 0-7


## p2b + det  r101 without pseudo
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_r101_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r101_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False   model.roi_head.bbox_head.with_loss_pseudo=False  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + 3stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=3 model.roi_head.bbox_head.num_stages=3  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py /home/ubuntu/mnt/dataset/MSCOCO2017/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'detection2/without_weight/epoch_12.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + 4stage
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_4stage/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=4 model.roi_head.bbox_head.num_stages=4  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py /home/ubuntu/mnt/dataset/MSCOCO2017/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

##test
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/test/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.0002 load_from='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_withoutneg/epoch_12.pth'

work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/test/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/cas_P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False model.roi_head.num_stages=3 model.roi_head.bbox_head.num_stages=3  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' optimizer.lr=0.0002 load_from='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_3stage/epoch_12.pth'

## p2b + det  without pseudo +topk k=3 7 10
sleep 5h
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_top3/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.roi_head.top_k=2 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2_14/without_weight' --resume-from ${work_dir}'detection2/without_weight/epoch_12.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo without neg + mil-1 in stage2
sleep 3h
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_withoutneg_mil1stage2/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.train_cfg.fine_proposal.gen_num_neg=0 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + mil-1 in stage2
sleep 3h
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_mil1stage2/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' model.train_cfg.fine_proposal.gen_num_neg=100 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7




## p2b + det  with pseudo 
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' load_from=${work_dir}'epoch_12.pth' evaluation.do_first_eval=True runner.max_epochs=0 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py /home/ubuntu/mnt/dataset/MSCOCO2017/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'detection2/without_weight/epoch_12.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

# p2b + det 1stage posi
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_onestage_softmax_posi/'&& 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  --resume-from ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.num_stages=1 model.roi_head.bbox_head.num_stages=1  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10002 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight'  --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' load_from=${work_dir}'epoch_12.pth' evaluation.do_first_eval=True runner.max_epochs=0 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco14.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


### calculate mean iou
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_r101_withoutpseudo/'
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' '../TOV_mmdetection_cache/work_dirs/center_like/COCO/test' 

## different detectors
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_top3/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2_retinanet/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_800_latest_pseudo_ann_1.json' dataset_type='CocoFmtDataset'
python exp/tools/killgpu.py 0-7 

#########
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_neg1000/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' model.train_cfg.fine_proposal.gen_num_neg=1000
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2_14/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

###ufo2
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ufo2_faster/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8  --resume-from ${work_dir}'detection2_retinanet/without_weight/epoch_2.pth' --work-dir=${work_dir}'detection2_retinanet/without_weight' --cfg-options data.train.ann_file='/home/ubuntu/hxm/wetectron/output/coco/test/instances_train2017_ufo2.json' 
python exp/tools/killgpu.py 0-7 


### ignore cut
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_ignorecut/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.train_cfg.base_proposal.cut_mode='ignore' evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --resume-from ${work_dir}'detection2/without_weight/epoch_10.pth' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


### no cut
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_nocut/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} --resume-from ${work_dir}'epoch_8.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


### noshare fc
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_nosharefc_softmax_sigmoidneg_milmil_withoutpseudo' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,TOV_mmdetection5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False model.roi_head.bbox_head.num_ref_fcs=2 evaluation.save_result_file=${work_dir}'_1200_latest_result.json' 
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7
TOV_mmdetection




work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_retrain/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.top_k=4 model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7 
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco14.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7








##rebuttal
#swin transformer
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/swin/swin_t_ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_swin-t-p4-w7_fpn_1x_coco.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco14.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo ++no jitter
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_nojitter/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7


## p2b + det  without pseudo + weighted_cls_topk
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/rebuttal/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_weighted_cls_topk/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 2 \
--work-dir=${work_dir}  --resume-from '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## vgg 16
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/rebuttal/vgg16/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_vgg16_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  --resume-from  ${work_dir}'epoch_12.pth' \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## vgg 19
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/rebuttal/vgg19/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_vgg19_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7




## p2b + det  without pseudo + LVIS
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/LVIS/rebuttal/ms_cas_sharefc_softmax_sigmoidneg_milmil_without/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/LVIS/P2BNet/P2BNet_r50_fpn_1x_lvis_ms.py 8 \
--work-dir=${work_dir}  --resume-from  ${work_dir}'epoch_4.pth'  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo +imbalance
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/rebuttal/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_imbalancek/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms_unbalance.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7

## p2b + det  without pseudo + x50
work_dir='../TOV_mmdetection_cache/work_dirs/center_like/COCO/rebuttal/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo_x50/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_x50_fpn_1x_coco_ms.py 8 \
--work-dir=${work_dir}  \
--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
python exp/tools/killgpu.py 0-7
python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ${work_dir}'_1200_latest_result.json' ${work_dir}'coco_1200_latest_pseudo_ann_1.json'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 --work-dir=${work_dir}'detection2/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
python exp/tools/killgpu.py 0-7
