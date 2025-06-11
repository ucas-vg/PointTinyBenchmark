_base_ = [
    'p2p_r50_fpn_1x_fl_sl1_coco400.py'
]

# dataset settings
data_root = 'data/coco/'

data = dict(
    train=dict(
        ann_file=data_root+'coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json'
    ),
)
