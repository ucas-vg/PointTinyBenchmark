_base_ = ['./P2SegDistill_r50_fpn_3x_voc_12SBD_ms.py']
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
