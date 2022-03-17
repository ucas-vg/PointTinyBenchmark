norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

_base_ = [
    'p2p_r50_fpns4_1x_fl_sl1_coco.py'
]

debug = False

# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(667, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(667, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline,
               ann_file=data_root+'coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json'),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

model = dict(
    neck=dict(
        start_level=1,  # 1     # output only 1 level (cause p2p has no size)
        num_outs=1,  # 5
    ),
    bbox_head=dict(strides=[8]),
)
