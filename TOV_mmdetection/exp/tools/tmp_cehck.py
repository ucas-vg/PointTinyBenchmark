import json
import math

jd = json.load(
    open(
        # 'data/tiny_set_v2/anns/release/corner/coarse/noise_rg-0-0.25_1/corner_w640_h640/pseuw16h16/rgb_train_w640h640ow100oh100_coarse.json',
        '../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r7_7_lr0.05_1x_2b4g//latest_result_refine2_2_r7_7.json'
    )
)

# for ann in jd['annotations']:
#     if math.isnan(ann['area']):
#         print(ann)

for res in jd:
    print(res)
    break