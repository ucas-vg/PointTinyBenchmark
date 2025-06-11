_base_ = ['./faster_rcnn_r50_fpn_3x_VOC_ms.py']


# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = 'data/'
data = dict(
    samples_per_gpu=1,  # 2
    workers_per_gpu=2,  # didi-debug 2
    train=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2012_SBD/cocostyle_coarse_annotations_new/quasi-center-point-0-0-0.25-0.25-0.25_1_with_truebox/qc_voc12sbd_ins_train_cls.json",
        img_prefix=data_root + 'VOC2012_SBD/JPEGImages',  # 'train2017/',
    ),
    val=dict(
        samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root + "VOC2012_SBD/cocostyle/voc12sbd_ins_val_cls.json",
        img_prefix=data_root + 'VOC2012_SBD/JPEGImages',  # 'train2017/',
        # min_gt_size=2
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)  # 4*GPU:LR0.004