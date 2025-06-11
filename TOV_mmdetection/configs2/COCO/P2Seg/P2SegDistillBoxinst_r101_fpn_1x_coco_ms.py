_base_ = [
    './P2SegDistillBoxinst_r50_fpn_1x_coco_ms.py'
]
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))