debug = True

# 1.1 train_pipeline:Resize; Collect(gt_true_bboxes);
# 1.2 test_pipeline: load annotation; scale_factor;
# 2. data: min_gt_size, train_ann(coarse), val_ann set as train_ann; test_mode
# 3. evaluation: maxDets

_base_ = [
    'coarse_point_refine_r50_fpn_1x_coco400.py'
]

dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(667, 400), keep_ratio=True),
    # dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
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
        img_scale=(667, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),  # add
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes']),  # gt_true_bboxes use for debug
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline,),
    test=dict(pipeline=test_pipeline)
)


model = dict(
    neck=dict(
        start_level=1,  # 1
        num_outs=1,  # 5
    ),
    bbox_head=dict(
        debug=debug,
        num_classes=80,  # 80
        strides=[8],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
    ),
)
