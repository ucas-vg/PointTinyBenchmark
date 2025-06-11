debug = False

# 1.1 train_pipeline:Resize; Collect(gt_true_bboxes);
# 1.2 test_pipeline: load annotation; scale_factor;
# 2. data: min_gt_size, train_ann(coarse), val_ann set as train_ann; test_mode
# 3. evaluation: maxDets

_base_ = [
    'coarse_point_refine_r50_fpns4_1x_coco400.py'
]

model = dict(
    neck=dict(
        start_level=1,  # 1
        num_outs=1,  # 5
    ),
    bbox_head=dict(
        num_classes=80,  # 80
        strides=[8],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
    ),
)
