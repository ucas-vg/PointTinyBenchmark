_base_ = ['./faster_rcnn_r50_fpn_3x_VOC12_ms.py']
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
