import os

configs = [
    {
        "host": "ubuntu@124.16.75.192",
        "root": "yxh/github/TOV_mmdetection_cache/work_dir/",
        "port": 33187
    }
]

local_root = "../TOV_mmdetection_cache/work_dir/"

paths = [
    # "TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr002_1x_4g",
    # "TinyPerson/Base/retinanet_r50_fpns4_1x_TinyPerson640/old640x512_lr0005_1x_1g",
    # "TinyPerson/Base/retinanet_r50_fpns4_1x_TinyPerson640/old640x512_lr0005_1x_1g_2",
    # "TinyPerson/Base/retinanet_r50_fpns4_1x_TinyPerson640/old640x512_lr002_1x_clipg_4g",
    # "TinyCOCO/coarsepoint/pt0.5x0.5y/pseuw16h16/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept/lr0.04_1x_16b4g/"
    "TinyCOCO/coarsepoint//noise_uniform_1//pseuw16h16/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept/lr0.04_1x_16b4g/round0",
    "TinyCOCO/coarsepoint//noise_uniform_1//pseuw16h16/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept/lr0.04_1x_16b4g/round1_aRp",
    "TinyCOCO/coarsepoint//noise_uniform_1//pseuw16h16/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept/lr0.04_1x_16b4g/round2_aRp-aRp",
    "TinyCOCO/coarsepoint//noise_uniform_1//pseuw16h16/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept/lr0.04_1x_16b4g/round3_aRp-aRp-aRp"
]

for cfg in configs:
    for path in paths:
        local_path = local_root + "/" + path
        # path_dir, _ = os.path.split(local_path)
        path_dir = local_path
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        cmd = f"scp -r -P{cfg['port']} {cfg['host']}:{cfg['root']}/{path}/ {local_path}"
        print(cmd)
        os.system(cmd)
