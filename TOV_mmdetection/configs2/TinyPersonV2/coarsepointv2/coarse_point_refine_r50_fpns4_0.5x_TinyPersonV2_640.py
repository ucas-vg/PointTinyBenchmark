_base_ = ['coarse_point_refine_r50_fpns4_1x_TinyPersonV2_640.py']


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)
