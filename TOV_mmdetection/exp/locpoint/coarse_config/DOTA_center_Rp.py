from .coarse_config import *
__all__ = ["config_class", "exps_config"]

# TinyCOCO
config_class = TinyCOCOConfig
net = {
    'aRepPoint': "configs2/DOTA/coarsepoint/reppoints_moment_r50_fpn_gn-neck+head_1x_DOTA_1024_centerpt.py",
}

com_cfg = dict(
    ann_root="data/dota/DOTA-split/trainsplit/",
    save_root="../TOV_mmdetection_cache/work_dir/DOTA/coarsepoint/",
    gpus=[0, 1],
    port=10002,
    presolve=("pseuw{}h{}", 64, 64),
    input_ann=LastAttr("output_ann"),
    noise_type="noise_rg-0-0-0.25-0.25_1",
    corner="",
    true_ann="data/dota/DOTA-split/trainsplit/DOTA_train1024.json",
    init_coarse_ann="data/dota/DOTA-split/trainsplit/center/pseuw64h64/DOTA_train1024_center.json",
    train_img_root="data/dota/DOTA-split/trainsplit/images",
    max_grow_rate=2.,
    cfg_options={
        "optimizer.lr": 0.005,
        "data.samples_per_gpu": 2,
    }
)  # equal to last output_ann

exps_config = [
    {"round_dir": "round0", "config_file": net['aRepPoint'],
     "input_ann": "data/dota/DOTA-split/trainsplit/center/pseuw64h64/DOTA_train1024_center.json"},
]

exps_config = get_exps_config(com_cfg, exps_config)

# python exp/locpoint/coarse_point_manager.py -c exp/locpoint/coarse_config/TinyCOCO_aRp_aRp_aRp_aRp.py
