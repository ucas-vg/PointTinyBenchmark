_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',
    # '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add
debug = False
# model settings
det_loss_weight = 4.0
stage_modes=['CBP', 'PBR']
num_stages = 2
model = dict(
    type='ENoiseBox',
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
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,  # 5
        norm_cfg=norm_cfg
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),  #
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=det_loss_weight), ),  # 1.0
    roi_head=dict(
        type='EP2BplusHead',
        num_stages=num_stages,
        stage_modes=stage_modes,
        top_k=7,
        with_atten=False,
        cluster_mode='mil_cls',  ###'cluster','upper'
        # stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head1=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=det_loss_weight),  # 1.0
            loss_bbox=dict(type='L1Loss', loss_weight=det_loss_weight)),  # 1.0
        bbox_head=dict(
            type='Shared2FCInstanceMILHeadEPLUS',
            num_stages=num_stages,
            stage_modes=stage_modes,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            num_ref_fcs=0,
            with_reg=True,
            with_sem=False,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            with_loss_pseudo=False,
            loss_p2b_weight=1.0,  # 7/19 dididi
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='binary_cross_entropy'),  # weight
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),  # weight
            # loss_re_cls=dict(type='FocalLoss', use_sigmoid=True,
            #                  gamma=2.0, alpha=0.25, loss_weight=1.0, activated=True),
            # loss_re_cls=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),  # 1.0
            # loss_re_cls=dict(
            #     type='MILLoss',
            #     binary_ins=False,
            #     loss_weight=0.25,
            #     loss_type='binary_cross_entropy'),  # weight
            # loss_bbox=dict(
            #     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_bbox_ori=dict(
                type='L1Loss', loss_weight=0.25),
            loss_bbox=dict(
                type='L1Loss', loss_weight=0.25),
        ),

    ),
    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            # base_ratios=[1 / 2, 1.0, 2.0],
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_per_scale=200,
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            # shake_ratio=([0.2], None),
            # # base_ratios_shake=[1.0, 1/1.5, 0.5, 1.5,2.0],
            # base_ratios=([1.0, 0.5, 0.7, 1.5, 2.0], [1.0, 0.7, 0.8, 1.2, 1.5]),
            shake_ratio=[0.2],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            gen_num_per_box=10,
            iou_thr=0.3,
            gen_num_neg=500,
        ),
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=debug)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #     dict(type='Resize', img_scale=(2000, 1200), keep_ratio=True),
    # dict(type='Resize', img_scale=(333, 200), keep_ratio=True),
    # dict(type='Resize', img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
    dict(type='Resize', img_scale=[(1333, 800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),
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
    samples_per_gpu=2,  # 2
    workers_per_gpu=2,  # didi-debug 2
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'noisy_pkl/instances_train2017_noise-r0.4.json',
        img_prefix=data_root + 'images/',  # 'train2017/',
        pipeline=train_pipeline,
        # min_gt_size=2
    ),
    val=dict(
        samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',  # 'train2017/',
        pipeline=test_pipeline,
        test_mode=True,  # modified  7/21 True for test-5000, False for test-4952
        # min_gt_size=2
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)  # add by hui

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
# load_from = '../TOV_mmdetection_cache/work_dirs/center_like/COCO/ms_cas_sharefc_softmax_sigmoidneg_milmil_withoutpseudo/epoch_12.pth'
# load_from = '/home/dj/didi/ssd-det/outputs/coco/epoch_12.pth'
# load_from=''
work_dir = '../TOV_mmdetection_cache/work_dirs/NoiseBox/COCO/noise_ann_with_pbr_2stage'

evaluation = dict(
    interval=1, metric='bbox',
    # save_result_file=work_dir + '_' + str(test_scale) + '_latest_result.json',
    do_first_eval=False,  # test
    do_final_eval=False,
)
# evaluation = dict(interval=1, metric='bbox', do_first_eval=True,do_final_eval=True)
find_unused_parameters = True
