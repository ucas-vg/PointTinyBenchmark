debug = False

# 1.1 train_pipeline:Resize; Collect(gt_true_bboxes);
# 1.2 test_pipeline: load annotation; scale_factor;
# 2. data: min_gt_size, train_ann(coarse), val_ann set as train_ann; test_mode
# 3. evaluation: maxDets

_base_ = [
    'coarse_point_refine_r50_fpn_1x_coco400.py'
]

model = dict(
    type='BasicLocator',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
    ),
)
