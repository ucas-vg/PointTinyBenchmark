norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

_base_ = [
    'p2p_r50_fpns4_1x_fl_sl1_TinyPersonV2_640.py',
]

lr_config = dict(
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

