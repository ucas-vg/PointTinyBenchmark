from .coarse_config import *
__all__ = ["config_class", "exps_config"]

config_class = TinyPersonConfig
net = {
    'aRepPoint': "configs2/TinyPersonV2/coarsepoint/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPersonV2_640_coarsept.py",
    'aRetinanet': "configs2/TinyPersonV2/coarsepoint/retinanet_r50_fpns4_1x_TinyPersonV2_640_coarsept.py",
    'aRetinanetA1': "configs2/TinyPersonV2/coarsepoint/retinanetA1_r50_fpns4_1x_TinyPersonV2_640_coarsept.py",
}

com_cfg = dict(
    ann_root="data/tiny_set_v2/anns/release/corner/coarse/",
    save_root="../TOV_mmdetection_cache/work_dir/TinyPersonV2/coarsepoint/",
    gpus=[0, 1, 2, 3],
    port=10002,
    presolve=("pseuw{}h{}", 16, 16),
    input_ann=LastAttr("output_ann"),
    noise_type="noise_rg-0-0.25_1",
    corner="corner_w640_h640",
    true_ann="data/tiny_set_v2/anns/release/corner/rgb_train_w640h640ow100oh100.json",
    init_coarse_ann=Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/rgb_train_w640h640ow100oh100_coarse.json"),
    train_img_root="data/tiny_set_v2/imgs/",
    max_grow_rate=1.75,
)  # equal to last output_ann
exps_config = [
    {"round_dir": "round0", "config_file": net['aRetinanetA1'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/rgb_train_w640h640ow100oh100_coarse.json")},
    {"round_dir": "round1_aReA1", "config_file": net['aRetinanetA1']},
    {"round_dir": "round2_aReA1-aReA1", "config_file": net['aRetinanetA1']},
]

exps_config = get_exps_config(com_cfg, exps_config)
