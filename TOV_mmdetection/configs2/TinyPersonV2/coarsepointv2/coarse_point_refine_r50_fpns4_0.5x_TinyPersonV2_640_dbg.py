debug = True
_base_ = ['coarse_point_refine_r50_fpns4_1x_TinyPersonV2_640.py']

dataset_type = 'CocoFmtDataset'
data_root = 'data/tiny_set_v2/'
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'corner')
         ),  # gt_true_bboxes use for debug
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
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'corner')
                 ),  # gt_true_bboxes use for debug
        ])
]
data = dict(
    shuffle=True,
    train=dict(
        pipeline=train_pipeline
    ),
    val=dict(
        pipeline=test_pipeline,
        test_mode=False  # modified
    ),
    test=dict(pipeline=test_pipeline)
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

model = dict(
    bbox_head=dict(debug=debug)
)
