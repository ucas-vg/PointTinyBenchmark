_base_ = [
    './mask_rcnn_r50_fpn_3x_VOC12SBD_ms.py'
]

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
