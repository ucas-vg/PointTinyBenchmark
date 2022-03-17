debug = False

# 1.1 train_pipeline:Resize; Collect(gt_true_bboxes);
# 1.2 test_pipeline: load annotation; scale_factor;
# 2. data: min_gt_size, train_ann(coarse), val_ann set as train_ann; test_mode
# 3. evaluation

_base_ = [
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py',
    '../../_base_/models/cpr/coarse_point_refine_r50_fpns4_1x.py'
]

dataset_type = 'CocoFmtDataset'
data_root = 'data/dota/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),  # gt_true_bboxes use for debug
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # add
    dict(
        type='MultiScaleFlipAug',
        # # img_scale=(1333, 800),
        # type='CroppedTilesFlipAug',
        # tile_shape=(640, 640),  # sub image size by cropped
        # tile_overlap=(100, 100),
        scale_factor=[1.0],

        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),  # add
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes']),  # gt_true_bboxes use for debug
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + 'DOTA-split/trainsplit/noise_rg-0-0-0.25-0.25_1/pseuw16h16/DOTA_train_1024_coarse.json',
        img_prefix=data_root + 'DOTA-split/trainsplit/images',
        pipeline=train_pipeline
    ),
    val=dict(
        min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + 'DOTA-split/trainsplit/noise_rg-0-0-0.25-0.25_1/pseuw16h16/DOTA_train_1024_coarse.json',
        img_prefix=data_root + 'DOTA-split/trainsplit/images',
        pipeline=test_pipeline,
        test_mode=False  # modified
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'anns/release/rgb_test.json',
        img_prefix=data_root + 'imgs/',
        pipeline=test_pipeline)
)

model = dict(
    neck=dict(
        start_level=0,  # 1
        num_outs=1,  # 5
    ),
    bbox_head=dict(
        num_classes=15,  # 80
        strides=[4],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
    ),
    # training and testing settings
    test_cfg=dict(
        nms_pre=2000,
        max_per_img=1000)
)

# location bbox eval
evaluation = dict(
    interval=13, metric='bbox',
    skip_eval=True,
    do_first_eval=True,
    do_final_eval=True,
    use_location_metric=True,
    location_kwargs=dict(
        class_wise=False,
        matcher_kwargs=dict(multi_match_not_false_alarm=False),
        location_param=dict(
            matchThs=[0.5, 1.0, 2.0],
            recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
            maxDets=[1000],
            # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
            # maxDets=[1000],
        )
    )
)

optimizer = dict(lr=0.01)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
find_unused_parameters = True
