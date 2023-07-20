import os


for e in range(1, 13):
    cmd = f"python tools/train.py configs2/COCO/coarsepointv2/coarse_point_refine_r50_fpn_1x_coco400_dbg.py \
    --work-dir ../TOV_mmdetection_cache/work_dir/tmp/ \
    --resume-from ../TOV_mmdetection_cache/work_dir/COCO/coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0_r8_8_lr0.01_1x_8b8g/epoch_{e}.pth \
    --cfg-options evaluation.do_first_eval=True \
      model.bbox_head.loss_cfg.with_gt_loss=True model.bbox_head.point_refiner.merge_th=0.1 \
      model.bbox_head.point_refiner.refine_th=0.1 model.bbox_head.point_refiner.classify_filter=True \
      model.bbox_head.refine_pts_extractor.pos_generator.radius=8 " \
      f"model.bbox_head.debug_info.epoch={e}"
    print(cmd)
    os.system(cmd)
