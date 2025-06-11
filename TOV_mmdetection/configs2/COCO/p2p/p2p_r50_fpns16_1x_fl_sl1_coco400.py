_base_ = [
    'p2p_r50_fpns4_1x_fl_sl1_coco400.py'
]

model = dict(
    neck=dict(
        start_level=2,  # 1     # output only 1 level (cause p2p has no size)
        num_outs=1,  # 5
    ),
    bbox_head=dict(strides=[16]),
)
