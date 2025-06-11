_base_ = ['./P2BNet_r50_fpn_1x_VOC_07_ms.py']

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=24, metric='bbox',
)
