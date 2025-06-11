from .coarse_config import *
__all__ = ["config_class", "exps_config"]

# TinyCOCO
config_class = TinyCOCOConfig
net = {
    'aRepPoint': "configs2/COCO/coarsepoint/reppoints_moment_r50_fpn_gn-neck+head_1x_coco400_coarsept.py",
    'Retina': "configs2/COCO/coarsepoint/retinanet_r50_fpn_1x_coco400_coarsept.py",
}

com_cfg = dict(
    ann_root="data/coco/coarse_gen_annotations/",
    save_root="../TOV_mmdetection_cache/work_dir/COCO/coarsepoint/",
    gpus=[0, 1, 2, 3, 4, 5, 6, 7],
    port=10002,
    presolve=("pseuw{}h{}", 64, 64),
    input_ann=LastAttr("output_ann"),
    noise_type="noise_rg-0-0-0.25-0.25_1",
    corner="",
    true_ann="data/coco/annotations/instances_train2017.json",
    init_coarse_ann=Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/instances_train2017_coarse.json"),
    train_img_root="data/coco/images/",
    max_grow_rate=2.,

    cfg_options={
        "optimizer.lr": 0.04,
        "data.samples_per_gpu": 8,
    }
)  # equal to last output_ann

exps_config = [
    {"round_dir": "round0", "config_file": net['Retina'],
     "input_ann": Var(lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/instances_train2017_coarse.json")},
    {"round_dir": "round1_re", "config_file": net['Retina']},
    {"round_dir": "round2_re-re", "config_file": net['Retina']},
    # {"round_dir": "round3_aRp-aRp-aRp", "config_file": net['aRepPoint']}
]

exps_config = get_exps_config(com_cfg, exps_config)

# python exp/locpoint/coarse_point_manager.py -c exp/locpoint/coarse_config/TinyCOCO_aRp_aRp_aRp_aRp.py
