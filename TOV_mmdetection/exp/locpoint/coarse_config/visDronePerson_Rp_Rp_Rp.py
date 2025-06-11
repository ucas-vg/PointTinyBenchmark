from .coarse_config import *
__all__ = ["config_class", "exps_config"]

# TinyCOCO
config_class = VisDronePersonConfig
net = {
    'RepPoint': "configs2/visDronePerson/coarsepoint/reppoints_moment_r50_fpn_gn-neck+head_1x_visDronePerson640_coarse.py",
}
com_cfg = {
    'gpus': [0, 1, 2, 3],
    'port': 10000,
    'presolve': ("pseuw{}h{}", 32, 32),
    "input_ann": LastAttr("output_ann")}  # equal to last output_ann
exps_config = [
    {"round_dir": "round0", "config_file": net['RepPoint'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/VisDrone2018-DET-train-person-w640h640ow100oh100_coarse.json")},
    {"round_dir": "round1_Rp", "config_file": net['RepPoint']},
    {"round_dir": "round2_Rp-Rp", "config_file": net['RepPoint']}
]

exps_config = get_exps_config(com_cfg, exps_config)
