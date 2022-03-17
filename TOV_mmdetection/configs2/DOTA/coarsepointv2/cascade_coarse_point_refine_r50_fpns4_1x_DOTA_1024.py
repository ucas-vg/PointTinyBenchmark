debug = False

# 1.1 train_pipeline:Resize; Collect(gt_true_bboxes);
# 1.2 test_pipeline: load annotation; scale_factor;
# 2. data: min_gt_size, train_ann(coarse), val_ann set as train_ann; test_mode
# 3. evaluation

_base_ = [
    'coarse_point_refine_r50_fpns4_1x_DOTA_1024.py'
]

num_stages = 2
model=dict(
    bbox_head=dict(
        type='CascadeCPRHead',
        cascade_cfg=dict(
            gt_src='gt_refine',
            weight_with_score=False,
            weight_type='max',
            conditional_refine=True,
            increase_r=False,
            increase_r_step=1,
        ),
        loss_cfg=dict(
            refine_bag_policy='only_refine_bag',
            with_gt_loss=True,
            gt_loss_type='gt',
        ),
        point_refiner=dict(
            merge_th=0.1,
            refine_th=0.1,
            classify_filter=True,
        ),
        train_pts_extractor=dict(
            pos_generator=dict(type='CirclePtFeatGenerator', radius=3),
            neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3, class_wise=True)),
        refine_pts_extractor=dict(
            pos_generator=dict(type='CirclePtFeatGenerator', radius=3),
            neg_generator=dict(type='OutCirclePtFeatGenerator', radius=3, keep_wh=True, class_wise=True),
        ),
        cpr_cfg_list=[dict(type='CPRHead') for r in range(3, 3 + num_stages)]),
)

