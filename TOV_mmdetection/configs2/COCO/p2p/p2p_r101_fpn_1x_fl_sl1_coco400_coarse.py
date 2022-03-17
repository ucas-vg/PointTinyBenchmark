_base_ = [
    'p2p_r50_fpn_1x_fl_sl1_coco400_coarse.py'
]

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
    ),
)

