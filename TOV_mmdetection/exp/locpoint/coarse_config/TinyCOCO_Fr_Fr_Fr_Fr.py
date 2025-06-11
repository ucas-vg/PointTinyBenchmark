from .coarse_config import *
__all__ = ["config_class", "exps_config"]

# TinyCOCO
config_class = TinyCOCOConfig
net = {
    'Rp': "configs2/TinyCOCO/coarsepoint/reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept.py",
    'Fr': "configs2/TinyCOCO/coarsepoint/faster_rcnn_r50_fpn_1x_tinycoco_corasept.py"
}
com_cfg = {"input_ann": LastAttr("output_ann"), 'gpus': [4, 5, 6, 7], 'port': 9999}  # equal to last output_ann
exps_config = [
    {"round_dir": "round0", "config_file": net['Fr'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/instances_train2017_100x167_coarse.json")},
    {"round_dir": "round1_Fr", "config_file": net['Fr']},
    {"round_dir": "round2_Fr-Fr", "config_file": net['Fr']},
    {"round_dir": "round2_Fr-Fr-Fr", "config_file": net['Fr']}
]

exps_config = get_exps_config(com_cfg, exps_config)

# python exp/locpoint/coarse_point_manager.py -c exp/locpoint/coarse_config/TinyCOCO_Fr_Fr_Fr_Fr.py
