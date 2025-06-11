_base_ = [
    'P2BNet_r50_fpn_1x_coco_ms.py',
    # '../_base_/datasets/coco_instance.py',
    # '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
fp16 = dict(loss_scale=dict(init_scale=512))
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='P2BNet',
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
    ),
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            # base_ratios=[1 / 2, 1.0, 2.0],
            shake_ratio=None,
            cut_mode='clamp',  # 'clamp',
            gen_num_per_scale=200,
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            # add_small_neg=True,
            # shake_ratio=([0.2], None),
            # # base_ratios_shake=[1.0, 1/1.5, 0.5, 1.5,2.0],
            # base_ratios=([1.0, 0.5, 0.7, 1.5, 2.0], [1.0, 0.7, 0.8, 1.2, 1.5]),
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            gen_num_per_box=10,
            iou_thr=0.7,
            gen_num_neg=500,
        ),
    ))
    # roi_head=dict(
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7),
    #         out_channels=256,
    #         featmap_strides=[8,16,32,64]),))

dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #     dict(type='Resize', img_scale=(2000, 1200), keep_ratio=True),
    # dict(type='Resize', img_scale=(333, 200), keep_ratio=True),
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]]),
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
        img_scale=(2000, test_scale) if test_scale else (1333, 800),
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
    samples_per_gpu=2,  # 2
    workers_per_gpu=2,  # didi-debug 2
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'pts_annotation_published/instances_train2017_refine2_2_r8_8.json',
        # ann_file='/home/pfchen/disk1/cpf/P2BNet/TOV_mmdetection/work_dirs/coco_imbalance_train.json',
        # ann_file=data_root + 'noisy_pkl/instances_train2017_noise-r0.4.json',
        ann_file=data_root + "coarse_gen_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/instances_train2017_coarse.json",
        # ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',  # 'train2017/',

        pipeline=train_pipeline,
        # min_gt_size=2
    ),
    val=dict(
        samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + "coarse_gen_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/instances_train2017_coarse.json",
        # ann_file=data_root + "resize/annotations/instances_train2017_100x167.json",
        img_prefix=data_root + 'images',  # 'train2017/',
        pipeline=test_pipeline,
        test_mode=False,  # modified
        # min_gt_size=2
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'roi_head.': dict(lr_mult=0.1)
        }))

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)
find_unused_parameters=True