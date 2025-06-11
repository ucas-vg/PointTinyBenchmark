_base_ = [
    'P2BNet_r50_fpn_1x_coco_ms.py',
    # '../_base_/datasets/coco_instance.py',
    # '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'  # noqa
#
# dict(_delete_=True,
#         type='mmpretrain.ConvNeXt',
#         arch='base',
#         out_indices=[0, 1, 2, 3],
#         # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
#         drop_path_rate=0.4,
#         layer_scale_init_value=0.,  # disable layer scale when using GRN
#         gap_before_final_norm=False,
#         use_grn=True,  # V2 uses GRN
#         init_cfg=dict(
#             type='Pretrained', checkpoint=checkpoint_file,
#             prefix='backbone.')),
#     neck=dict(in_channels=[128, 256, 512, 1024]),

model = dict(
    type='P2BNet',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4,8,16,32]),),
    train_cfg = dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            # base_ratios=[1 / 2, 1.0, 2.0],
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_per_scale=200,
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            # add_small_neg=True,
            # shake_ratio=([0.2], None),
            # # base_ratios_shake=[1.0, 1/1.5, 0.5, 1.5,2.0],
            # base_ratios=([1.0, 0.5, 0.7, 1.5, 2.0], [1.0, 0.7, 0.8, 1.2, 1.5]),
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            gen_num_per_box=10,
            iou_thr=0.7,
            gen_num_neg=500,
        ),
))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # 'backbone': dict(lr_mult=0.1)
        }))


lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)
# find_unused_parameters=True