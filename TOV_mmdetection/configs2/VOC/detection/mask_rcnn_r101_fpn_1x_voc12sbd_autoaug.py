_base_ = ['./mask_rcnn_r101_fpn_3x_VOC12SBD_ms.py']
# dataset settings
dataset_type = 'CocoFmtDataset'
# data_root = '/cache/'
data_root = 'data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=False),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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

test_pipeline_aug = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1000, 600), (1800, 1000)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data_train_12 = dict(
#     type='VOCDatasetInstance',
#     ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_12.txt',
#     img_prefix=data_root + 'VOC2012/',
#     pipeline=train_pipeline
# )
# data_train_sbd= dict(
#     type='SBDDatasetInstance',
#     ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_sbd.txt',
#     img_prefix=data_root + 'VOC2012/',
#     pipeline=train_pipeline
# )
# data_train_12 = dict(
#     type='RepeatDataset',
#     times=3,
#     dataset=dict(
#         type='VOCDatasetInstance',
#         ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_12.txt',
#         img_prefix=data_root + 'VOC2012/',
#         pipeline=train_pipeline
#     )
# )
# data_train_sbd= dict(
#     type='RepeatDataset',
#     times=3,
#     dataset=dict(
#         type='SBDDatasetInstance',
#         ann_file=data_root + 'VOC2012/ImageSets/Segmentation/train_aug_sbd.txt',
#         img_prefix=data_root + 'VOC2012/',
#         pipeline=train_pipeline
#     )
# )

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file= data_root+'VOC2012_SBD/cocostyle/voc12sbd_ins_train_cls.json',
            img_prefix=data_root + 'VOC2012_SBD/JPEGImages',
            pipeline=train_pipeline
        ),
        pipeline=None
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012_SBD/cocostyle/voc12sbd_ins_val_cls.json',
        img_prefix=data_root + 'VOC2012_SBD/JPEGImages',
        pipeline=test_pipeline_aug),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012_SBD/cocostyle/voc12sbd_ins_val_cls.json',
        img_prefix=data_root + 'VOC2012_SBD/JPEGImages',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='mAP_Segm')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(interval=1, metric=['bbox', 'segm'])

