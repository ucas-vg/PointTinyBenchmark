_base_ = [
    '../base/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
]

debug = False
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Resize', img_scale=(333, 200), keep_ratio=True),
    dict(type='Resize', img_scale=(667, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # img_scale=(333, 200),
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

dataset_type = 'CocoFmtDataset'
data = dict(
    samples_per_gpu=2,
    train=dict(
        type=dataset_type, pipeline=train_pipeline,
        ann_file=data_root+"coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw32h32/instances_train2017_coarse.json",
        img_prefix=data_root + 'images/',  # 'train2017/',),
    ),
    val=dict(
        type=dataset_type, pipeline=test_pipeline,
        ann_file=data_root+"annotations/instances_val2017.json",
        img_prefix=data_root + 'images/',
    ),
    test=dict(
        type=dataset_type, pipeline=test_pipeline,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + 'images/',
    ),
)

# location bbox eval
evaluation = dict(
    interval=1, metric='bbox',
    use_location_metric=True,
    location_kwargs=dict(
        class_wise=False,
        matcher_kwargs=dict(multi_match_not_false_alarm=False),
        location_param=dict(
            matchThs=[0.5, 1.0, 2.0],
            recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
            maxDets=[100],
            # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
            # maxDets=[1000],
        )
    )
)

