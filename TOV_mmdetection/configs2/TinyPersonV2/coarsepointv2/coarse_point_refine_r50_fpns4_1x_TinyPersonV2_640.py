norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

_base_ = ['coarse_point_refine_base_TinyPersonV2_640.py']

debug = False
alpha = 0.25  # 0.25
model = dict(
    type='BasicLocator',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,  # 1
        add_extra_convs='on_input',
        num_outs=1,  # 5
        norm_cfg=norm_cfg,  # add
    ),
    bbox_head=dict(
        norm_cfg=norm_cfg,  # add
        type='CPRHead',
        # type='CascadeCPRHead',
        num_classes=1,  # 80
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        num_cls_fcs=0,
        strides=[4],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]

        loss_mil=dict(
            type='MILLoss',
            binary_ins=False,
            loss_weight=alpha),  # weight
        loss_type=0,
        loss_cfg=dict(
            with_neg=True,
            neg_loss_weight=1-alpha,
            refine_bag_policy='independent_with_gt_bag',
            random_remove_rate=0.4,
            with_gt_loss=True,
            gt_loss_weight=alpha,
            with_mil_loss=True,
        ),
        normal_cfg=dict(
            prob_cls_type='sigmoid',
            out_bg_cls=False,
        ),
        train_pts_extractor=dict(
            pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
            neg_generator=dict(type='OutCirclePtFeatGenerator', radius=5, class_wise=True),
        ),
        refine_pts_extractor=dict(
            pos_generator=dict(type='CirclePtFeatGenerator', radius=5),
            # neg_generator=dict(type='AnchorPtFeatGenerator', scale_factor=1.0),
            neg_generator=dict(type='OutCirclePtFeatGenerator', radius=5, keep_wh=True, class_wise=True),
        ),
        point_refiner=dict(
            merge_th=0.1,
            refine_th=0.1,
            classify_filter=True,

        ),
    ),
    # training and testing settings
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1000))

find_unused_parameters = True

