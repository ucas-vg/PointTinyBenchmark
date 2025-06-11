_base_ = [
    './mask_rcnn_r50_fpn_1x_voc12sbd_mmseg.py'
]

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
