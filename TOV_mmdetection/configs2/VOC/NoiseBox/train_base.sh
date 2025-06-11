##E-noiseBox
#N=0.4 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc/ \
# --resume-from=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc/latest.pth

#### top k
#K=4 && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}k${K}/ \
# --cfg-options  optimizer.lr=${LR} model.roi_head.top_k=${K}

### iou_thr # 0.1 0.5
#IOU=0.1 && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}iou${IOU}/ \
# --cfg-options  optimizer.lr=${LR} model.train_cfg.fine_proposal.iou_thr=${IOU}

#IOU=0.5 && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}iou${IOU}/ \
# --cfg-options  optimizer.lr=${LR} model.train_cfg.fine_proposal.iou_thr=${IOU}
#python exp/tools/killgpu.py 0-1


#### gen_num_neg # 300 1000
#NeN=300 && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}neg${NeN}/ \
# --cfg-options  optimizer.lr=${LR} model.train_cfg.fine_proposal.gen_num_neg=${NeN}
#python exp/tools/killgpu.py 0-1

#### base_ratios # [1, 1.25, 1.5, 0.75, 0.5][1, 1.5, 2.0, 0.75, 0.5]
#BRa=[1,1.5,2.0,0.75,0.5] && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}_brat2/ \
# --cfg-options  optimizer.lr=${LR} model.train_cfg.fine_proposal.base_ratios=${BRa}
#python exp/tools/killgpu.py 0-1

# ### shake_ratio=[0.2],  # 0.1 0.3 0.4
#SRa=0.1 && N=0.4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc_lr${LR}_srat${SRa}/ \
# --cfg-options  optimizer.lr=${LR} model.train_cfg.fine_proposal.shake_ratio=[${SRa}]
#python exp/tools/killgpu.py 0-1
