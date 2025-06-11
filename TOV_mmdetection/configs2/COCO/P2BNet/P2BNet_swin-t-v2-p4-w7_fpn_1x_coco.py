_base_ = [
    'P2BNet_r50_fpn_1x_coco_ms.py',
    # '../_base_/datasets/coco_instance.py',
    # '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# fp16 = dict(loss_scale=dict(init_scale=512))
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
# pretrained='https://github.com/SwinTransformer/storage/releases/tag/v1.0.8/swin_tiny_patch4_window7_224_22k.pth'
pretrained='swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth'


# model = dict(backbone=dict(window_size=[16, 16, 16, 8]))
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='SwinTransformerV2',
#         arch='base',
#         window_size=[16, 16, 16, 8],
#         img_size=256,
#         drop_path_rate=0.5),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=1000,
#         in_channels=1024,
#         init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
#         loss=dict(
#             type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
#         cal_acc=False),
#     init_cfg=[
#         dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
#         dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
#     ],
#     train_cfg=dict(augments=[
#         dict(type='Mixup', alpha=0.8),
#         dict(type='CutMix', alpha=1.0)
#     ]),
# )
model = dict(
    type='P2BNet',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV2',
        arch='tiny',
        window_size=[16, 16, 16, 8],
        img_size=256,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),),
    # backbone=dict(
    #     _delete_=True,
    #     type='SwinTransformer',
    #     embed_dims=96,
    #     depths=[2, 2, 6, 2],
    #     num_heads=[3, 6, 12, 24],
    #     window_size=7,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.2,
    #     patch_norm=True,
    #     out_indices=(0, 1, 2, 3),
    #     with_cp=False,
    #     convert_weights=True,
    #     init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    # ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
    ),
    roi_head=dict(
        top_k=7,
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
            iou_thr=0.3,
            gen_num_neg=1000,
        ),
))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'roi_head.': dict(lr_mult=0.1)
        }))


lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)
# find_unused_parameters=True