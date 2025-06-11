

LR=0.001 && GPU=4 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 tools/dist_train.sh \
  configs2/TinyCOCO/coarsepointv2/coarse_point_refine_r50_fpns4_1x_tinycoco.py $GPU   \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyCOCO/coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_tinycoco/f4_neg_loss0_r5_lr${LR}_1x_16b4g \
  --cfg-options optimizer.lr=${LR} model.neck.num_outs=1 model.bbox_head.strides=[4] model.bbox_head.feature_extractor.neighbour_cfg.neighbour_radius=5 model.bbox_head.loss_mil.type='MIL2Loss'

LR=0.001 && GPU=4 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 tools/dist_train.sh \
  configs2/TinyCOCO/coarsepointv2/coarse_point_refine_r50_fpns4_1x_tinycoco.py $GPU   \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyCOCO/coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_tinycoco/f4_loss0_r5_lr${LR}_1x_16b4g \
  --cfg-options optimizer.lr=${LR} model.neck.num_outs=1 model.bbox_head.strides=[4] model.bbox_head.loss_mil.type='MIL2Loss' \
  model.bbox_head.feature_extractor.neighbour_cfg.neighbour_radius=5 \
  model.bbox_head.feature_extractor.neg_cfg.scale=0.0

