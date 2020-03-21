# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import torchvision.transforms as TT
import numpy as np
MT = T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if cfg.INPUT.USE_SCALE:
        resize = MT.ScaleResize(cfg.INPUT.SCALES, cfg.INPUT.SCALE_MODE)
    else:
        resize = T.Resize(min_size, max_size)

    transform = T.Compose(
        [
            resize,
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    # ############################################# add by hui ##############################################
    assert not cfg.DATALOADER.USE_SCALE_MATCH or cfg.DATALOADER.USE_MORE_DA == 4, 'current only DA4 support SCALE_MATCH.'
    if is_train and cfg.DATALOADER.USE_MORE_DA == 2:
        min_crop_overlap = cfg.DATALOADER.DA_MIN_CROP_OVERLAP if len(cfg.DATALOADER.DA_MIN_CROP_OVERLAP) > 0 else None
        gt_range = cfg.DATALOADER.DA_WANT_GT_RANGE if len(cfg.DATALOADER.DA_WANT_GT_RANGE) > 0 else None

        crop_size = cfg.DATALOADER.DA_CROP_SIZE
        min_size, max_size = min(crop_size), max(crop_size)
        transform = T.Compose(
            [
                # MT.RandomExpand(max_ratio=1.25, fill=cfg.INPUT.PIXEL_MEAN, keep_ratio=False, transform_prob=0.5),
                MT.ImageToImageTargetTransform(
                    TT.ColorJitter(brightness=32 / 255, contrast=0.5, saturation=0.5, hue=0.1),
                    transform_prob=0.5),
                MT.RandomCropResizeForBBox2(gt_range, crop_size,
                                            fill=cfg.INPUT.PIXEL_MEAN, scale_range=cfg.DATALOADER.DA_GT_SCALE_RANGE,
                                            min_crop_size_ratio=cfg.DATALOADER.DA_MIN_CROP_SIZE_RATIO,
                                            min_crop_overlap=min_crop_overlap,
                                            constraint_auto=True, transform_prob=cfg.DATALOADER.DA_CROP_RESIZE_PROB),
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif is_train and cfg.DATALOADER.USE_MORE_DA == 3:
        min_crop_overlap = cfg.DATALOADER.DA_MIN_CROP_OVERLAP if len(cfg.DATALOADER.DA_MIN_CROP_OVERLAP) > 0 else None
        gt_range = cfg.DATALOADER.DA_WANT_GT_RANGE if len(cfg.DATALOADER.DA_WANT_GT_RANGE) > 0 else None
        transform = T.Compose(
            [
                # MT.RandomExpand(max_ratio=1.25, fill=cfg.INPUT.PIXEL_MEAN, keep_ratio=False, transform_prob=0.5),
                MT.ImageToImageTargetTransform(
                    TT.ColorJitter(brightness=32 / 255, contrast=0.5, saturation=0.5, hue=0.1),
                    transform_prob=0.5),
                # MT.RandomCropResizeForBBox2((3.5, 30), (min_size, max_size), fill=cfg.INPUT.PIXEL_MEAN,
                #                             scale_range=(1./4, 4.), min_crop_size_ratio=0.5/8, constraint_auto=True,
                #                             transform_prob=0.5),
                MT.RandomCropResizeForBBox3(gt_range, cfg.DATALOADER.DA_CROP_SIZE,
                                            fill=cfg.INPUT.PIXEL_MEAN, scale_range=cfg.DATALOADER.DA_GT_SCALE_RANGE,
                                            min_crop_size_ratio=cfg.DATALOADER.DA_MIN_CROP_SIZE_RATIO,
                                            min_crop_overlap=min_crop_overlap,
                                            constraint_auto=True, transform_prob=cfg.DATALOADER.DA_CROP_RESIZE_PROB),
                # T.Resize(min_size, max_size),
                resize,
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    elif is_train and cfg.DATALOADER.USE_MORE_DA == 4:
        transform = []

        # color aug
        if cfg.DATALOADER.DA4_COLOR_AUG:
            transform.append(MT.ImageToImageTargetTransform(
                TT.ColorJitter(brightness=32 / 255, contrast=0.5, saturation=0.5, hue=0.1), transform_prob=0.5))

        # scale aug
        scale_range = cfg.DATALOADER.DA4_SCALE_RANGE
        scales = cfg.DATALOADER.DA4_SCALES
        if len(scale_range) > 0:
            assert len(scale_range) == 2 and len(scales) == 0, \
                'DA4_SCALE_RANGE and DA4_SCALES can only specified one of them.'
            transform.append(MT.RandomScaleResize(scale_range[0], scale_range[1], cfg.INPUT.SCALE_MODE))
        else:
            assert len(scales) > 0
            transform.append(MT.ScaleResize(scales, cfg.INPUT.SCALE_MODE))

        # translate aug
        offset_x_range = cfg.DATALOADER.DA4_OFFSET_X_RANGE
        offset_y_range = cfg.DATALOADER.DA4_OFFSET_Y_RANGE
        if len(offset_x_range) == 2:
            xmin, xmax = offset_x_range
            ymin, ymax = offset_y_range
            # transforms.append(MT.Translate(offset_x_range, offset_y_range))
            l, r, t, b = -xmin, xmax, -ymin, ymax
            blue, green, red = cfg.INPUT.PIXEL_MEAN
            transform.append(MT.RandomCrop(size=None, padding=(l, t, r, b),
                                           fill=np.array([red, green, blue]).astype(np.int)))

        if cfg.DATALOADER.USE_SCALE_MATCH:
            resize = MT.ScaleMatchFactory.create(cfg.DATALOADER.SCALE_MATCH)

        transform.extend(
            [
                resize,
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        transform = T.Compose(transform)
    return transform
