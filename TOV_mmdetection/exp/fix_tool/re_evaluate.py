import os

sub_dirs1 = [
    # "c1c_loss0gt_r3_3_lr0.01_1x_16b4g",     # 39.22 => 39.48
    # "c2c_loss0gt_r3_3_lr0.01_1x_16b4g",     # 38.93 => 39.20
    # "c2c_ws_loss0gt_r3_3_lr0.01_1x_16b4g",  # 38.92 => 39.10
    # "loss0gt_2ins_r3_3_lr0.01_1x_16b4g",    # 37.81 => 37.98
    # "loss0gt_allpos_r3_3_lr0.01_1x_16b4g",  # 30.49 => 30.59
    # "loss0gt_drop0.0_r3_3_lr0.01_1x_16b4g", # 38.03 => 38.18
    # "loss0gt_drop0.6_r3_3_lr0.01_1x_16b4g", # 37.77 => 37.98
    # "loss0gt_nomil_r3_3_lr0.01_1x_16b4g",   # 35.78 => 36.06
    # "loss0gt_noneg_r3_3_lr0.01_1x_16b4g",   # 29.64 => 29.79
    # "loss0gt_r1_1_lr0.01_1x_16b4g",         # 31.48 => 31.67
    # "loss0gt_r2_2_lr0.01_1x_16b4g",         # 35.81 => 35.98
    # "loss0gt_r3_3_NormSig_lr0.01_1x_16b4g", # 31.40 => 31.58
    # "loss0gt_r3_3_bgneg_lr0.01_1x_16b4g",   # 37.68 => 37.94
    # "loss0gt_r4_4_lr0.01_1x_16b4g",         # 39.00 => 39.19
    # "loss0gt_r5_3_lr0.01_1x_16b4g",         # 37.03 => 37.21
    # "loss0gt_r6_6_lr0.01_1x_16b4g",         # 39.67 => 39.87
    # "loss0gt_r7_7_lr0.01_1x_16b4g",         # 39.75 => 39.98
    # "",                                     # 28.39 => 28.49
    # "pseudo_box_refine_round1_aRp",           # => 35.35
    # "pseudo_box_refine_round3_aRp-aRp-aRp",   # => 33.82
]

# sub_dirs2 = [
#     "loss0_r3_3_lr0.01_1x_16b4g",
#     "loss0_r5_3_lr0.01_1x_16b4g",
#     "loss0_r8_3_lr0.01_1x_16b4g",
# ]

# sub_dirs3 = [
#     "loss0gt_r3_3_lr0.01_1x_16b4g",    # 37.94
#     "loss0gt_r5_5_lr0.01_1x_16b4g",    # 39.97
# ]

# for sub_dir in sorted(os.listdir("../TOV_mmdetection_cache/work_dir/TinyCOCO/p2p_coarse/noise_rg-0-0-0.25-0.25_1")):
#     print('"' + sub_dir + '",')

old_root_dir = '../TOV_mmdetection_cache/work_dir/TinyCOCO/p2p_coarse_old/noise_rg-0-0-0.25-0.25_1'
root_dir = '../TOV_mmdetection_cache/work_dir/TinyCOCO/p2p_coarse/noise_rg-0-0-0.25-0.25_1'

for d in sub_dirs1:
    old_d = os.path.join(old_root_dir, d, 'p2p_r50_fpns4_1x_fl_sl1_tinycoco/adam0.0001_1x_16b4g')
    files = list(os.listdir(old_d))
    assert len([f for f in files if f.endswith('.log')]) == 1, old_d

    new_d = os.path.join(root_dir, d, 'p2p_r50_fpns4_1x_fl_sl1_tinycoco/adam0.0001_1x_16b4g')
    os.makedirs(new_d)
    cmd = f'cp {old_d}/epoch_12.pth {new_d}/ && ln -s epoch_12.pth {new_d}/latest.pth && ' \
          f'export LR=0.0001 && export BATCH=16 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 ' \
          f'tools/dist_train.sh {old_d}/p2p_r50_fpns4_1x_fl_sl1_tinycoco.py 4 --resume-from {new_d}/latest.pth' \
          f' --cfg-options evaluation.do_final_eval=True'
    print(cmd)
    os.system(cmd)

old_root_dir = '../TOV_mmdetection_cache/work_dir/TinyCOCO'
old_d = os.path.join(old_root_dir, 'p2p/p2p_r50_fpns4_1x_fl_sl1_tinycoco/adam1e-4_1x_16b4g')
cmd = 'export LR=0.0001 && export BATCH=16 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 ' \
     f'tools/dist_train.sh {old_d}/p2p_r50_fpns4_1x_fl_sl1_tinycoco.py 4 --resume-from {old_d}/latest.pth' \
     f' --cfg-options evaluation.do_final_eval=True'
print(cmd)
os.system(cmd)