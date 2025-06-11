_base_ = 'P2BNet_swin-t-p4-w7_fpn_1x_coco.py'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
pretrained='/home/ubuntu/cpf/P2BNet/TOV_mmdetection/swin_small_patch4_window7_224_22k.pth'# noqa
model = dict(
    pretrained=None,
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    roi_head=dict(
        top_k=7),
    train_cfg = dict(
        base_proposal=dict(
            base_scales=[ 4,8, 16, 32, 64, 128],
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
            gen_num_neg=500,
        ),
))
# fp16 = dict(loss_scale=dict(init_scale=512))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'roi_head': dict(lr_mult=0.1)
        }))


lr_config = dict(warmup_iters=500, step=[8, 11])
runner = dict(max_epochs=12)
# find_unused_parameters=True