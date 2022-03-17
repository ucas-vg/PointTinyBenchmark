norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

_base_ = [
    '../../_base_/datasets/TinyPersonV2/TinyPersonV2_detection_640x640.py',
    '../../../configs/_base_/default_runtime.py'
]

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
        num_classes=1,
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
        nms_pre=2000,   # 1000
        min_bbox_size=0,
        score_thr=0.05,
        pseudo_wh=(16, 16),
        nms=dict(type='nms', iou_threshold=0.2),
        max_per_img=1000))

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
# tiny bbox eval with IOD
evaluation = dict(
    interval=3, metric='bbox',
    use_location_metric=True,
    location_kwargs=dict(
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
