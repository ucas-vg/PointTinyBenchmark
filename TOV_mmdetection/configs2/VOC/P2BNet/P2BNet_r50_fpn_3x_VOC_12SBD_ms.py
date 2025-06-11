_base_ = ['./P2BNet_r50_fpn_1x_VOC_12SBD_ms.py']

lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(
    interval=36, metric='bbox',
)