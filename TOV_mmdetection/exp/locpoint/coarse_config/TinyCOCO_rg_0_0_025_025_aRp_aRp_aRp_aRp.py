from .coarse_config import *
__all__ = ["config_class", "exps_config"]

# TinyCOCO
config_class = TinyCOCOConfig
net = {
    'aRepPoint': "configs2/TinyCOCO/coarsepoint/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept.py",
}
com_cfg = {
    'gpus': [0, 1, 2, 3],
    'port': 10002,
    'presolve': ("pseuw{}h{}", 16, 16),
    "input_ann": LastAttr("output_ann"),
    "noise_type": "noise_rg-0-0-0.25-0.25_1"
}  # equal to last output_ann
exps_config = [
    {"round_dir": "round0", "config_file": net['aRepPoint'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/instances_train2017_100x167_coarse.json")},
    {"round_dir": "round1_aRp", "config_file": net['aRepPoint']},
    {"round_dir": "round2_aRp-aRp", "config_file": net['aRepPoint']},
    {"round_dir": "round3_aRp-aRp-aRp", "config_file": net['aRepPoint']}
]

exps_config = get_exps_config(com_cfg, exps_config)

# python exp/locpoint/coarse_point_manager.py -c exp/locpoint/coarse_config/TinyCOCO_aRp_aRp_aRp_aRp.py
