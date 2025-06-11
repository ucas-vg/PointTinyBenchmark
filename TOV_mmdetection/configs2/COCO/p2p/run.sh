# LR, B = 2
# stride, input size

# input:400, stride: 8/4/16
GPU=8 && LR=0.0001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns4_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns4_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# input:800, stride: 8/4
GPU=8 && LR=0.0001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns4_1x_fl_sl1_coco.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns4_1x_fl_sl1_coco/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# stride=16, B=4/8, LR=0.0002/0.0004 (linear principle, but not work for adam)
GPU=8 && LR=0.0002 && B=4 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0004 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# stride=16, LR=1e-4, Batch=4/8/12/16, adjust Batch
GPU=8 && LR=0.0001 && B=1 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=12 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=16 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# stride=16, LR=5e-5, Batch=8, adjust LR
GPU=8 && LR=0.00005 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# stride 8, LR=1e-4, Batch=8
GPU=8 && LR=0.0001 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=16 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

# LR=1e-3/1e-5
#GPU=8 && LR=0.001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400.py ${GPU} \
#        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
#        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}
#GPU=8 && LR=0.00001 && B=2 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400.py ${GPU} \
#        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400/adam${LR}_1x_${B}b${GPU}g${V}/ \
#        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}


# coarse point
# stride=16/8
GPU=8 && LR=0.0001 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400_coarse.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpns16_1x_fl_sl1_coco400_coarse/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

GPU=8 && LR=0.0001 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse.py ${GPU} \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse/adam${LR}_1x_${B}b${GPU}g${V}/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}

