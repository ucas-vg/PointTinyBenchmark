_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',
    # '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add
debug = False
# model settings
stage_modes=['CBP', 'PBR','PBR']

num_stages = 3
model = dict(
    type='P2Seg',
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
        bbox_head=dict(
            type='Shared2FCInstanceMILHeadEPLUS',
            num_stages=num_stages,
            stage_modes=stage_modes,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            num_ref_fcs=0,
            with_reg=True,
            with_sem=False,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            with_loss_pseudo=False,
            with_others=True,
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
            loss_bbox_ori=dict(
                type='L1Loss', loss_weight=0.25),
            loss_bbox=dict(
                type='L1Loss', loss_weight=0.25),
        ),

    ),
    bbox_head=dict(
        type='CondInstBoxHead',
        num_classes=20,
        in_channels=256,
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    mask_branch=None,
    mask_head=None,

    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            # base_ratios=[1 / 2, 1.0, 2.0],
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_per_scale=None,
            gen_num_neg=None),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.1],
            base_ratios=[1, 1.1, 1.2, 1.3, 0.9, 0.8, 0.7],
            gen_num_per_box=10,
            iou_thr=0.3,
            gen_num_neg=500,
        ),
        rcnn=dict()),
    test_cfg=dict(
        rpn=dict(),
        rcnn=dict()))

# dataset settings
# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #     dict(type='Resize', img_scale=(2000, 1200), keep_ratio=True),
    # dict(type='Resize', img_scale=(333, 200), keep_ratio=True),
    dict(type='Resize', img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),
]

test_scale = 1200
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # add
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # img_scale=(333, 200),
        img_scale=(2000, test_scale) if test_scale else (1333, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=1,  # 2
    workers_per_gpu=2,  # didi-debug 2
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'pts_annotation_published/instances_train2017_refine2_2_r8_8.json',
        # ann_file=data_root + 'pts_annotation_published/instances_train_val_2017_point.json',
        ann_file=data_root + "VOC2007/Annotations-QC-0-0-0.25-0.25-0.25_coco_fmt/voc07_trainval.json",
        # ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root,  # 'train2017/',

        pipeline=train_pipeline,
        # min_gt_size=2
    ),
    val=dict(
        samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + "VOC2007/Annotations-QC-0-0-0.25-0.25-0.25_coco_fmt/voc07_trainval.json",
        # ann_file=data_root + "resize/annotations/instances_train2017_100x167.json",
        img_prefix=data_root,  # 'train2017/',
        pipeline=test_pipeline,
        test_mode=False,  # modified
        # min_gt_size=2
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))


check = dict(stop_while_nan=False)  # add by hui

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=4)
find_unused_parameters = True
# load_from = '../TOV_mmdetection_cache/work_dirs/P2Seg/P2SegDistillBoxInst_new/epoch_12.pth'
# load_from = '../TOV_mmdetection_cache/work_dirs/P2Seg/P2BBoxInst/epoch_12.pth'
work_dir = '../TOV_mmdetection_cache/work_dirs/P2Seg/P2BBoxInst/'

evaluation = dict(
    interval=36, metric='bbox',
    do_first_eval=False,  # test
    do_final_eval=True,
    # save_result_file=f'{work_dir}/latest_resul.json',
)
