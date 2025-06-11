from .coarse_config import *
__all__ = ["config_class", "exps_config"]

config_class = TinyPersonConfig
net = {
    'aRepPoint': "configs2/TinyPerson/coarsepoint/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_coarsept.py"
}
com_cfg = {"input_ann": LastAttr("output_ann")}  # equal to last output_ann
exps_config = [
    {"round_dir": "round0", "config_file": net['aRepPoint'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json")},
    {"round_dir": "round1_aRp", "config_file": net['aRepPoint']},
    {"round_dir": "round2_aRp-aRp", "config_file": net['aRepPoint']},
]

exps_config = get_exps_config(com_cfg, exps_config)
