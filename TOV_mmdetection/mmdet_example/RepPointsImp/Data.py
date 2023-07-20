if __name__ == '__main__':
    # block1: build dataloader ########################################################################
    from mmdet.datasets.builder import DATASETS
    from mmdet.datasets import build_dataloader, build_dataset

    data_root = '/home/hui/dataset/tiny_set/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        #     dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
        dict(type='Resize', scale_factor=[0.5], keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),  # 定义输出
    ]

    cfg = dict(
        type='CocoFmtDataset',
        ann_file=data_root + 'mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
        img_prefix=data_root + 'erase_with_uncertain_dataset/train/',
        pipeline=train_pipeline
    )

    # dataset = DATASETS.build(cfg=cfg)
    dataset = build_dataset(cfg)
    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=1, num_gpus=1,
                                  dist=False, shuffle=True, seed=None)  # invoke DataLoader

    # block2: print data of batch ############################################################

    gid = 0  # gpu_id
    img_id = 0
    for i, data_batch in enumerate(dataloader):
        print(len(data_batch), data_batch.keys())
        #     print(data_batch['img_metas'])
        # DataContainer --.data--> list(len=1) --[gpu_id]--> tensor(B, C, H, W)
        print('img:\t', type(data_batch['img']), type(data_batch['img'].data), len(data_batch['img'].data),
              type(data_batch['img'].data[gid]), data_batch['img'].data[gid].shape)
        print('gt_bboxes:\t', len(data_batch['gt_bboxes'].data), len(data_batch['gt_bboxes'].data[gid]),
              data_batch['gt_bboxes'].data[gid][img_id].shape)
        print()
        for k in data_batch:
            print(f'{k}:\t', data_batch[k].data[gid][img_id])
            print()
        break

    # ##############################################################
    # from setting import model as model_cfg
    # from mmdet.models.builder import DETECTORS
    #
    # class MyDict(dict):
    #     @staticmethod
    #     def to(d: dict):
    #         for k, v in d.items():
    #             if isinstance(v, dict) and not isinstance(v, MyDict):
    #                 d[k] = MyDict.to(v)
    #         return MyDict(d)
    #
    #     def __getattr__(self, k):
    #         return self[k]
    #
    #
    # model_cfg = MyDict.to(model_cfg)
    # # model_cfg.pop('pretrained')
    # ori_model = DETECTORS.build(model_cfg)


def build_data_loader():
    # block1: build dataloader ########################################################################
    from mmdet.datasets.builder import DATASETS
    from mmdet.datasets import build_dataloader, build_dataset

    data_root = 'D:\\mapping\\TOV_mmdetection\\data\\tiny_set\\'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        #     dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
        dict(type='Resize', scale_factor=[0.5], keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),  # 定义输出
    ]

    cfg = dict(
        type='CocoFmtDataset',
        ann_file=data_root + 'mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
        img_prefix=data_root + 'erase_with_uncertain_dataset/train/',
        pipeline=train_pipeline
    )

    # dataset = DATASETS.build(cfg=cfg)
    dataset = build_dataset(cfg)
    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=1, num_gpus=1,
                                  dist=False, shuffle=True, seed=None)  # invoke DataLoader
    return dataloader


# 创建一个Pseudo输入
import torch
def gen_pseudo_inputs(B, C, H, W, num_levels, device):
    feats = [torch.randn(B, C, H//(2**l), W//(2**l)).to(device) for l in range(num_levels)]
    gt_bboxes = [torch.rand(2, 4).to(device) * 400 for _ in range(B)]
    for i, bbox in enumerate(gt_bboxes):
        gt_bboxes[i][:, 2:] = gt_bboxes[i][:, :2] + gt_bboxes[i][:, 2:] / 10
    gt_bboxes_ignore = [torch.empty(0, 4).to(device) for _ in range(B)]
    gt_labels = [torch.zeros(2).to(device) for _ in range(B)]
    img_metas = [{"pad_shape": (H*32, W*32, 3)} for _ in range(B)]
    return feats, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore
