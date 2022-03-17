norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

_base_ = [
    '../../../configs/_base_/default_runtime.py'
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
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
    samples_per_gpu=8,  # 2
    workers_per_gpu=1,  # didi-debug 2
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        min_gt_size=2
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)  # add by hui

model = dict(
    type='BasicLocator',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # default stride=(4, 8, 16, 32)
        # with_max_pool=False,    # or first_conv_stride=1    # expect fpn stride to begin with 2 (instead of 4)
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,  # 1     # output only 1 level (cause p2p has no size)
        add_extra_convs='on_input',
        num_outs=1,  # 5
        norm_cfg=norm_cfg,  # add
    ),
    bbox_head=dict(
        norm_cfg=norm_cfg,  # add
        type='P2PHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,  # didi4 TODO: 论文中“three stacked convolutions”
        strides=[4],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
        # point_anchor=[(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)],  # Grid
        point_anchor=[(0., 0.)],  # Grid
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_reg=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
        pts_gamma=1,  # γ to 100.
        reg_norm=1,
    ),
    # training and testing settings
    train_cfg=dict(
        # pos_weight=-1,  # reppoints
        neg_weight=1.0,  # paper,λ_1: 0.5 #didi added
        assigner=dict(  # didi added
            type='HungarianAssignerV2',
            # cls_cost=dict(type='PointsProbCost', weight=2.0),
            # reg_cost=dict(type='PointsDisCost', weight=5e-2, p=2),  # paper,τ: 5e-2s
            cls_costs=dict(type='FocalLossCost', weight=2.0),
            # cls_costs=dict(type='ZeroCost'),
            reg_costs=dict(type='DisCostV2', weight=0.1, norm_with_img_wh=False),
            topk_k=5
        ),
        sampler=dict(type='PseudoSampler'),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        pseudo_wh=(32, 32),
        nms=dict(type='nms', iou_threshold=0.01),
        max_per_img=100))

# optimizer = dict(lr=1e-4)  # 0.01
# # optimizer = dict(type='Adam', lr=0.0005)
# # optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])


# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# location bbox eval
evaluation = dict(
    interval=3, metric='bbox',
    do_first_eval=False,  # test
    # do_final_eval=True,
    use_location_metric=True,
    location_kwargs=dict(
        class_wise=False,
        matcher_kwargs=dict(multi_match_not_false_alarm=False),
        location_param=dict(
            matchThs=[0.5, 1.0, 2.0],
            recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
            maxDets=[100],
            areaRng=[[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]],
            areaRngLbl=['all', 'small', 'medium', 'large']
        )
    )
)
