# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# ################################### add by hui ###################################
from maskrcnn_benchmark.structures.bounding_box import BoxList
from math import floor, ceil
import copy
# from MyPackage.tools.debug_log.debug_log import Logger
import os, time
import numbers
from .scale_match import *
# logger = Logger('ERROR')
PIL_RESIZE_MODE = {'bilinear': Image.BILINEAR, 'nearest': Image.NEAREST}


class ScaleResize(object):
    def __init__(self, scales, mode='bilinear'):
        if not isinstance(scales, (list, tuple)):
            scales = (scales,)
        self.scales = scales
        self.mode = PIL_RESIZE_MODE[mode]

    def __call__(self, image, target):
        origin_size = (image.width, image.height)
        scale = random.choice(self.scales)
        size = round(scale * image.height), round(scale * image.width)
        image = F.resize(image, size, self.mode)
        if scale == 1:
            assert image.size == origin_size, (image.size, size, image.width, image.height)
        target = target.resize(image.size)
        return image, target


class RandomScaleResize(object):
    def __init__(self, min_scale, max_scale, mode='bilinear'):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = PIL_RESIZE_MODE[mode]

    def __call__(self, image, target):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        size = round(scale * image.height), round(scale * image.width)
        image = F.resize(image, size, self.mode)
        target = target.resize(image.size)
        return image, target


class RandomCrop(torchvision.transforms.RandomCrop):
    def __init__(self, size=None, padding=0, pad_if_needed=False, max_try=3, fill=0):
        """
        :param size: (h, w), a square crop (size, size) is
            made. size=None means keep input image size, same as tranlate
        :param padding: (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        :param pad_if_needed: It will pad the image if smaller than the
            desired size to avoid raising an exception.
        """
        super(RandomCrop, self).__init__(size, padding, pad_if_needed)
        if isinstance(self.padding, numbers.Number):
            self.padding = (self.padding, self.padding, self.padding, self.padding)
        self.max_try = max_try
        self.not_success_time = 0
        self.fill = fill

    def __call__(self, img, target: BoxList):
        """
        Args:
            img (PIL Image): Image to be cropped.
            target:

        Returns:
            PIL Image: Cropped image.
        """
        if self.size is None:
            size = (img.size[1], img.size[0])  # (h, w)
        else:
            size = self.size

        if sum(self.padding) > 0:
            img = F.pad(img, self.padding, fill=tuple(self.fill))
            target = target.translate(self.padding[0], self.padding[1], clip=False)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < size[1]:
            img = F.pad(img, (int((1 + size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < size[0]:
            img = F.pad(img, (0, int((1 + size[0] - img.size[1]) / 2)))

        success = False
        for _ in range(self.max_try):
            i, j, h, w = self.get_params(img, size)
            tmp_target = target.translate(-j, -i, clip=True)
            if len(tmp_target.bbox) > 0:
                target = tmp_target
                success = True
                break
        if not success:
            i, j, h, w = (self.padding[1], self.padding[0], size[0], size[1])  # (t, l, h, w)
            target = target.translate(-j, -i, clip=True)
            self.not_success_time += 1
            if self.not_success_time % 100 == 0:
                warnings.warn("translate in RandomCrop, failed for {} times".format(self.not_success_time))

        return F.crop(img, i, j, h, w), target


# class Translate(object):
#     def __init__(self, offset_x_range, offset_y_range):
#         self.offset_x_range = list(offset_x_range)
#         self.offset_y_range = list(offset_y_range)
#
#     def get_offset_range(self, image:Image, target: BoxList):
#         def cal_offset_range(offset_x_range, x1, x2, w):
#             offset_x_range = copy.deepcopy(offset_x_range)
#             offset_y_range = copy.deepcopy(offset_y_range)
#             b1, b2 = x2.argmin(), x1.argmax()
#             offset_x_range[0] = max(offset_x_range[0], x2[b1] - w)
#             offset_x_range[1] = min(offset_x_range[1], x1[b2])
#             offset_y_range[0] =
#             return offset_x_range
#
#         bboxes = copy.deepcopy(target.bbox)
#         if target.mode == 'xywh':
#             bboxes[:, 2] += bboxes[:, 0]
#             bboxes[:, 3] += bboxes[:, 1]
#
#         b1 = np.argmin(bboxes[:, 0])
#
#         b2 = np.argmax(bboxes[:, 1])
#
#         if np.random.uniform(0, 1) > 0.5:
#             offset_x_range = cal_offset_range(self.offset_x_range, bboxes[:, 0], bboxes[:, 2], image.width)
#             offset_y_range = self.offset_y_range[:]
#         else:
#             offset_x_range = self.offset_x_range
#             offset_y_range = cal_offset_range(self.offset_y_range, bboxes[:, 1], bboxes[:, 3], image.height)
#         return offset_x_range, offset_y_range
#
#     def __call__(self, image: Image, target: BoxList):
#         offset_x_range, offset_y_range = self.get_offset_range(image, target)
#         offset_x = int(np.random.uniform(offset_x_range[0], offset_x_range[1]))
#         offset_y = int(np.random.uniform(offset_y_range[0], offset_y_range[1]))
#         image = Image.fromarray(self.translate(np.array(image), (offset_x, offset_y)))
#         target = target.translate(offset_x, offset_y)
#         return image, target
#
#     def translate(self, image, offset):
#         def do_offset(offset_x, w):
#             if offset_x < 0:
#                 sx1, sx2 = -offset_x, w
#                 dx1, dx2 = 0, w + offset_x
#             else:
#                 sx1, sx2 = 0, w - offset_x
#                 dx1, dx2 = offset_x, w
#             return (sx1, sx2), (dx1, dx2)
#
#         offset_x, offset_y = offset
#         h, w = image.shape[:2]
#         (sx1, sx2), (dx1, dx2) = do_offset(offset_x, w)
#         (sy1, sy2), (dy1, dy2) = do_offset(offset_y, h)
#         dst = np.zeros_like(image)
#         dst[dx1:dx2, dy1:dy2] = image[sx1:sx2, sy1:sy2]
#         return dst


class ImageToImageTargetTransform(object):
    def __init__(self, image_transform, transform_prob=1.):
        self.image_transform = image_transform
        self.transform_prob = transform_prob

    def __call__(self, image, target):
        if np.random.uniform(0, 1) > self.transform_prob:                   # whether use expand
            return image, target
        image = self.image_transform(image)
        return image, target


class RandomExpand(object):
    """
    random_expand: change from gluoncv gluoncv/data/transforms/image.py:220

    """
    def __init__(self, max_ratio=4, fill=0, keep_ratio=True, transform_prob=1.):
        if not isinstance(fill, np.ndarray):
            fill = np.array(fill).reshape((-1,))
        assert len(fill.shape) == 1 and (len(fill) == 1 or len(fill) == 3), 'fill shape must be (1,) or (3,).'
        if len(fill) == 1 and len(fill.shape) == 1: fill = np.array([fill, fill, fill])
        b, g, r = fill
        self.fill = np.array([r, g, b])
        self.max_ratio = max_ratio
        self.keep_ratio = keep_ratio
        self.transform_prob = transform_prob

    def __call__(self, image: Image.Image, target: BoxList):
        if np.random.uniform(0, 1) > self.transform_prob:                   # whether use expand
            return image, target
        mode, image = image.mode, np.array(image).astype(np.float32)
        image, (offset_x, offset_y, new_width, new_height) = self.random_expand(
            image, self.max_ratio, self.fill, self.keep_ratio)
        image = Image.fromarray(image.astype(np.uint8), mode)
        target = BoxList(target.bbox, (new_width, new_height), target.mode)   # size changed
        target = target.translate(offset_x, offset_y)                         # box translate
        return image, target

    def random_expand(self, src, max_ratio=4, fill=torch.Tensor([0, 0, 0]), keep_ratio=True):
        """Random expand original image with borders, this is identical to placing
        the original image on a larger canvas.

        Parameters
        ----------
        src : numpy.array
            The original image with HWC format.
        max_ratio : int or float
            Maximum ratio of the output image on both direction(vertical and horizontal)
        fill : int or float or array-like
            The value(s) for padded borders. If `fill` is numerical type, RGB channels
            will be padded with single value. Otherwise `fill` must have same length
            as image channels, which resulted in padding with per-channel values.
        keep_ratio : bool
            If `True`, will keep output image the same aspect ratio as input.

        Returns
        -------
        mxnet.nd.NDArray
            Augmented image.
        tuple
            Tuple of (offset_x, offset_y, new_width, new_height)

        """
        if max_ratio <= 1:
            return src, (0, 0, src.shape[1], src.shape[0])

        h, w, c = src.shape
        ratio_x = random.uniform(1, max_ratio)
        if keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, max_ratio)

        oh, ow = int(h * ratio_y), int(w * ratio_x)
        off_y = random.randint(0, oh - h)
        off_x = random.randint(0, ow - w)

        # make canvas
        dst = np.tile(fill, (oh, ow, 1))
        dst[off_y:off_y + h, off_x:off_x + w, :] = src
        return dst, (off_x, off_y, ow, oh)


class RandomCropResizeForBBox(object):
    """
            should combine with Resize(min_size, max_size)
    constrain:
        bbox constrain : at least have a gt box;
        scale constrain: 1. gt box scale to wanted [bbox_size_range]
                         2. cover at least [min_crop_size_ratio]^2 of origin image
                         3. scale must between [scale_range]
        translate      : cover at least [min_crop_size_ratio]^2 of origin image
    Method1:
        1. try getting a scale s that keep all gt boxes in origin image in [bbox_size_range] and s in [scale_range]
            -> choose_scale
        2. get crop's width and height (new_image_w, new_image_h) to origin image r.s.t scale s.
        3. try getting a crop that cover one of (leftest, topest, rightest, most bottom) gt box at least,
            and crop's left up point must left and up to center of origin image.  -> choose_crop
    Method2:
        1. random choose one of gt boxes B
        2. get a scale s to keep B size in [bbox_size_range] and s in [scale_range], get crop's width and height
        3. get a crop start_point that cover B and cover [min_crop_size_ratio]^2 of origin image at least

    in crop_image and crop_bbox
    add image scale for scale < 1 to prevent allocate a big memory dst in crop image, and speed up for big image
    """
    def __init__(self, bbox_size_range, crop_size_before_scale=None, fill=0, scale_range=(0, inf), min_crop_size_ratio=0.5,
                 min_crop_overlap=None, transform_prob=1.):
        self.bbox_size_range = bbox_size_range
        self.crop_size_before_scale = crop_size_before_scale  # None mean use input image size as crop_size_before_scale
        self.transform_prob = transform_prob
        if not isinstance(fill, np.ndarray):
            fill = np.array(fill).reshape((-1,))
        assert len(fill.shape) == 1 and (len(fill) == 1 or len(fill) == 3), 'fill shape must be (1,) or (3,).'
        if len(fill) == 1 and len(fill.shape) == 1: fill = np.array([fill, fill, fill]).reshape((-1,))
        b, g, r = fill
        self.fill = np.array([r, g, b])
        self.min_crop_size_ratio = min_crop_size_ratio
        self.scale_range = scale_range
        self.MAX_GT_WH = 200

        self.min_crop_overlap = min_crop_overlap
        assert min_crop_overlap is None or self.min_crop_size_ratio < 0

    def __call__(self, image: Image.Image, target: BoxList):
        """
        1. image gt box size is (s1 <= size <=s2), we choose a scale [s] r.s.t uniform(bbox_size_range[0]/s1, bbox_size_range[1]/s2)
           to make sure all gt box's size in image is: bbox_size_range[0] <= s * s1 <= s * s2 <= bbox_size_range[1].
        2. cal new_image's width and height respect tp scale [s].
        3. set origin image set axi(left-top is (0, 0)), choose a crop(new_image_x1, new_image_y1, new_image_x2, new_image_y2)
            respect ro new_image's width and height and include a gt box at least.
        4. move and crop annotation
        :param image:
        :param target:
        :return:
        """
        # print('image w, h', image.width, image.height)
        if np.random.uniform(0, 1) > self.transform_prob:                   # whether use expand
            return image, target
        old_image, old_target = copy.deepcopy(image), copy.deepcopy(target)
        try:
            # filter ignore and too big gt out, just to choose a scale and crop, final return will keep it.
            # TODO: should be replace with other policy to remove most ignore.
            boxes = target.bbox.cpu().numpy()
            non_ignore_boxes = np.all([boxes[:, 2] - boxes[:, 0] < self.MAX_GT_WH,
                                       boxes[:, 3] - boxes[:, 1] < self.MAX_GT_WH], axis=0)
            boxes = boxes[non_ignore_boxes]
            if len(boxes) == 0: return old_image, old_target

            # choose a scale and a crop r.s.t the scale
            scale = self.choose_scale(boxes)
            if scale is None: scale = 1.
            # choose a crop
            if self.crop_size_before_scale is None:
                crop_w_before_scale, crop_h_before_scale = image.width, image.height
            else:
                crop_w_before_scale, crop_h_before_scale = self.crop_size_before_scale
            crop = self.choose_crop(crop_w_before_scale, crop_h_before_scale,
                                    scale, boxes)  # crop can out of origin image
            if crop is None: return old_image, old_target
            # print(scale, crop, crop[2]-crop[0], crop[3]-crop[1])

            # crop bbox and image r.s.t choosed crop
            image = self.crop_image(image, crop, scale if scale < 1 else None)
            target = self.crop_bbox(target, crop, image.size)
        except BaseException as e:
            print(e)
            warnings.warn("exception happened which should not happened, may be some bug in code.")
            return old_image, old_target
        return image, target

    def choose_scale(self, boxes: np.ndarray):
        min_scale, max_scale = self.scale_range
        if self.bbox_size_range is not None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sizes = areas ** 0.5
            min_size, max_size = sizes.min(), sizes.max()
            max_scale = self.bbox_size_range[1] / max_size
            min_scale = self.bbox_size_range[0] / min_size
            min_scale = max(min_scale, self.scale_range[0])
            max_scale = min(max_scale, self.scale_range[1])
            if min_scale >= max_scale:
                # warnings.warn('RandomPadScaleForBBox failed, min_scale{} >= max_scale{}, cause bbox size({}, {})'
                #              ' variances are too big.'.format(min_scale, max_scale, min_size, max_size))
                return None
        scale = np.random.random() * (max_scale - min_scale) + min_scale
        # print('sizes', min_size, max_size, min_scale, max_scale, scale)
        return scale

    def choose_crop(self, crop_w_before_scale, crop_h_before_scale, scale, boxes: np.ndarray):
        crop_w_after_scale, crop_h_after_scale = crop_w_before_scale / scale, crop_h_before_scale / scale
        # crop can out of origin image, but must contain >=1 gt boxes.
        prob = np.random.random()
        if prob < 0.25:   contain_bbox = boxes[np.argmin(boxes[:, 0])]  # left box
        elif prob < 0.5:  contain_bbox = boxes[np.argmin(boxes[:, 1])]  # upper box
        elif prob < 0.75: contain_bbox = boxes[np.argmax(boxes[:, 2])]  # right box
        else:             contain_bbox = boxes[np.argmax(boxes[:, 3])]  # bottom box
        min_image_x1, min_image_y1 = contain_bbox[2] - crop_w_after_scale, contain_bbox[3] - crop_h_after_scale
        max_image_x1, max_image_y1 = contain_bbox[0], contain_bbox[1]

        # crop area must cover min_crop_size_ratio^2 of origin image, to avoid too least foreground info.
        if self.min_crop_size_ratio > 0:
            min_w, min_h = crop_w_after_scale * self.min_crop_size_ratio, crop_h_after_scale * self.min_crop_size_ratio
        else:
            min_w, min_h = np.array(self.min_crop_overlap) / scale
        min_image_x1 = max(min_image_x1, min_w - crop_w_after_scale)
        min_image_y1 = max(min_image_y1, min_h - crop_h_after_scale)
        max_image_x1 = min(max_image_x1, min_w)
        max_image_y1 = min(max_image_y1, min_h)
        if min_image_x1 >= max_image_x1 or min_image_y1 >= max_image_y1:
            # warnings.warn('RandomPadScaleForBBox failed, no crop available can find for scale.')
            return None

        # random choose a crop available
        image_x1 = int(np.random.random() * (max_image_x1 - min_image_x1) + min_image_x1)
        image_y1 = int(np.random.random() * (max_image_y1 - min_image_y1) + min_image_y1)
        image_x2 = int(image_x1 + crop_w_after_scale)
        image_y2 = int(image_y1 + crop_h_after_scale)
        crop = (image_x1, image_y1, image_x2, image_y2)
        # print(boxes, min_image_x1, max_image_x1, min_image_y1, max_image_y1)
        return crop

    def crop_image(self, image, crop, scale=None):
        image_width, image_height = image.width, image.height
        if scale is not None:
            crop = (np.array(crop) * scale).astype(np.int32)
            image_width, image_height = int(image_width * scale), int(image_height * scale)
            image = image.resize((image_width, image_height))

        image_x1, image_y1, image_x2, image_y2 = crop
        new_image_w, new_image_h = image_x2 - image_x1, image_y2 - image_y1
        if image_y1 < 0: dst_off_y, src_off_y = -image_y1, 0
        else: dst_off_y, src_off_y = 0, image_y1
        if image_x1 < 0: dst_off_x, src_off_x = -image_x1, 0
        else: dst_off_x, src_off_x = 0, image_x1
        if image_x2 > image_width:
            dst_off_x2, src_off_x2 = image_width - image_x1, image_width
        else:
            dst_off_x2, src_off_x2 = new_image_w, image_x2
        if image_y2 > image_height:
            dst_off_y2, src_off_y2 = image_height - image_y1, image_height
        else:
            dst_off_y2, src_off_y2 = new_image_h, image_y2

        src = np.array(image).astype(np.float32)
        dst = np.tile(self.fill, (new_image_h, new_image_w, 1))
        dst[dst_off_y:dst_off_y2, dst_off_x:dst_off_x2, :] = src[src_off_y:src_off_y2, src_off_x:src_off_x2, :]
        dst = dst.astype(np.uint8)
        # assert dst.shape[0] > 0 and dst.shape[1] > 0
        # try:
        image = Image.fromarray(dst, image.mode)
        # except OverflowError as e:
        #     print(e)
        #     print(dst)
        #     print(dst.max(), dst.min(), dst.shape, crop, image.mode)
        # print('image size', image.size)
        return image

    def crop_bbox(self, target, crop, image_size=None):
        _old_target = copy.deepcopy(target)
        image_x1, image_y1, image_x2, image_y2 = crop
        bboxes = target.convert('xyxy').bbox
        # translate(-image_x1, -image_y1)
        bboxes[:, 0] += -image_x1
        bboxes[:, 1] += -image_y1
        bboxes[:, 2] += -image_x1
        bboxes[:, 3] += -image_y1
        # crop(image_x1, image_y1, image_x2, image_y2)
        bboxes[:, 0] = bboxes[:, 0].clamp(0, image_x2-image_x1)
        bboxes[:, 1] = bboxes[:, 1].clamp(0, image_y2-image_y1)
        bboxes[:, 2] = bboxes[:, 2].clamp(0, image_x2-image_x1)
        bboxes[:, 3] = bboxes[:, 3].clamp(0, image_y2-image_y1)
        old_target = target
        target = BoxList(bboxes, (image_x2-image_x1, image_y2-image_y1), 'xyxy').convert(target.mode)
        target._copy_extra_fields(old_target)
        target = target.clip_to_image()
        assert len(target.extra_fields['labels']) == len(target.bbox)
        assert len(target.bbox) != 0, (crop, _old_target.bbox)
        if image_size is not None:
            target = target.resize(image_size)
        return target


class RandomCropResizeForBBox2(RandomCropResizeForBBox):
    """
            should combine with Resize(min_size, max_size)
    constrain:
        bbox constrain : at least have a gt box;
        scale constrain: 1. gt box scale to wanted [bbox_size_range]
                         2. cover at least [min_crop_size_ratio]^2 of origin image
                         3. scale must between [scale_range]
        translate      : cover at least [min_crop_size_ratio]^2 of origin image
    Method1:
        1. try getting a scale s that keep all gt boxes in origin image in [bbox_size_range] and s in [scale_range]
            -> choose_scale
        2. get crop's width and height (new_image_w, new_image_h) to origin image r.s.t scale s.
        3. try getting a crop that cover one of (leftest, topest, rightest, most bottom) gt box at least,
            and crop's left up point must left and up to center of origin image.  -> choose_crop
    Method2:
        1. random choose one of gt boxes B
        2. get a scale s to keep B size in [bbox_size_range] and s in [scale_range], get crop's width and height
        3. get a crop start_point that cover B and cover [min_crop_size_ratio]^2 of origin image at least
    """
    def __init__(self, bbox_size_range, crop_size_before_scale, fill=0, scale_range=(0, inf),
                 min_crop_size_ratio=0.5, min_crop_overlap=None, scale_constraint_type='all', crop_constrain_type='all',
                 constraint_auto=False, transform_prob=1., info_collector=None):
        super().__init__(bbox_size_range, crop_size_before_scale, fill, scale_range, min_crop_size_ratio,
                         min_crop_overlap, transform_prob)
        self.scale_constraint_type = scale_constraint_type
        self.crop_constrain_type = crop_constrain_type
        self.constraint_auto = constraint_auto
        if self.constraint_auto and (scale_constraint_type is not None or crop_constrain_type is not None):
            warnings.warn("constrain_auto set to True, scale_constraint_type and crop_constrain_type"
                          " will not use again.")
        self.info_collector = info_collector

        if self.bbox_size_range is not None and self.bbox_size_range[1] is None:
            self.bbox_size_range = (self.bbox_size_range[0], inf)
        if self.scale_range[1] is None:
            self.scale_range = (self.scale_range[0], inf)

    def __call__(self, image: Image.Image, target: BoxList):
        """
        should combine with Resize(min_size, max_size)
        1. image gt box size is (s1 <= size <=s2), we choose a scale [s] r.s.t uniform(bbox_size_range[0]/s1, bbox_size_range[1]/s2)
           to make sure all gt box's size in image is: bbox_size_range[0] <= s * s1 <= s * s2 <= bbox_size_range[1].
        2. cal new_image's width and height respect tp scale [s].
        3. set origin image set axi(left-top is (0, 0)), choose a crop(new_image_x1, new_image_y1, new_image_x2, new_image_y2)
            respect ro new_image's width and height and include a gt box at least.
        4. move and crop annotation
        :param image:
        :param target:
        :return:
        """
        if self.info_collector is not None:
            self._analysis_info = {'function_call':
                              {'deal_background': 0, 'crop_constraint_type=="all"': 0, 'crop_constraint_type=="one"': 0,
                               'scale_constraint_type=="all"': 0, 'scale_constraint_type=="one"': 0,
                               'scale_constraint_type=="mean"': 0, 'success': 0, 'crop_no_scale_constraint': 0}}

        # print('image w, h', image.width, image.height)
        if np.random.uniform(0, 1) > self.transform_prob:                   # whether use expand
            return self.deal_background(image, target)
        old_image, old_target = copy.deepcopy(image), copy.deepcopy(target)
        if True:
        # try:
            # 1. filter ignore and too big gt out, just to choose a scale and crop, final return will keep it.
            # TODO: should be replace with other policy to remove most ignore.
            boxes = target.bbox.cpu().numpy()
            # non_ignore_boxes = np.all([boxes[:, 2] - boxes[:, 0] < self.MAX_GT_WH,
            #                            boxes[:, 3] - boxes[:, 1] < self.MAX_GT_WH], axis=0)
            # boxes = boxes[non_ignore_boxes]
            # if len(boxes) == 0:
            #     logger("BG1, no non-ignore gt boxes found, boxes is {}".format(target.bbox.cpu().numpy()), 'DEBUG')
            #     return self.deal_background(old_image, old_target)

            # 2. choose boxes as constraint
            boxes = self.choose_boxes(boxes)

            # 3. choose a scale and a crop r.s.t the scale
            if self.constraint_auto:  # constraint_auto will try ['all', 'mean', 'one'] one by one.
                for constraint_type in ['all', 'mean', 'one']:
                    scale = self.choose_scale(boxes, constraint_type)
                    if scale is not None:
                        break
            else:
                scale = self.choose_scale(boxes, self.scale_constraint_type)

            if scale is None:
                return self.deal_background(old_image, old_target)

            # 4. choose a crop
            if self.crop_size_before_scale is None:
                crop_w_before_scale, crop_h_before_scale = image.width, image.height
            else:
                crop_w_before_scale, crop_h_before_scale = self.crop_size_before_scale

            if self.constraint_auto:
                for constrain_type in ['all', 'one']:
                    crop = self.choose_crop(crop_w_before_scale, crop_h_before_scale, image.width, image.height,
                                            scale, boxes, constrain_type)
                    if crop is not None:
                        break
            else:
                crop = self.choose_crop(crop_w_before_scale, crop_h_before_scale, image.width, image.height,
                                        scale, boxes, self.crop_constrain_type)  # crop can out of origin image

            if crop is None:
                return self.deal_background(old_image, old_target)
            # print(scale, crop, crop[2]-crop[0], crop[3]-crop[1])

            # 5. crop bbox and image r.s.t choose crop
            image = self.crop_image(image, crop, scale if scale < 1 else None)
            target = self.crop_bbox(target, crop, image.size)
            # if target is None:
                # return self.deal_background(old_image, old_target)


        # except BaseException as e:
        #     # print(e)
        #     # warnings.warn("exception happened which should not happened, may be some bug in code.")
        #     # raise e
        #     return self.deal_background(old_image, old_target)
        # print(crop[2]-crop[0], crop[3]-crop[1], image.size)

        if self.info_collector is not None:
            self._analysis_info['function_call']['success'] += 1
            self.info_collector(self._analysis_info)
        return image, target

    def deal_background(self, image, target):
        """ random crop image """
        if len(target.bbox) > 0:
            result = self.crop_with_no_scale_constraint(image, target)
            if result is not None:
                image, target = result

        if self.info_collector is not None:
            self._analysis_info['function_call']['deal_background'] += 1
            self.info_collector(self._analysis_info)
        return image, target

    def choose_boxes(self, boxes: np.array, choose_bbox_count=None):
        if choose_bbox_count is None:
            choose_bbox_count = np.random.randint(len(boxes)) + 1
        permutation = np.array(range(len(boxes)))
        np.random.shuffle(permutation)
        choose_box_idxes = permutation[:choose_bbox_count]
        return boxes[choose_box_idxes].copy()

    def choose_scale(self, boxes: np.ndarray, constraint_type='all'):
        """
        :param boxes:
        :param constraint_type: option in ["one", "mean", "all"],
                "one" means choose a scale let at least one of boxes' size scale to [bbox_size_range]
                "mean" means choose a scale let mean size of all boxes scale to [bbox_size_range]
                "all" means choose a scale let size of all boxes scale to [bbox_size_range]
        :return:
        """
        if self.info_collector is not None:
            self._analysis_info['function_call']['scale_constraint_type=="{}"'.format(constraint_type)] += 1
        min_scale, max_scale = self.scale_range
        if self.bbox_size_range is not None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sizes = areas ** 0.5
            if constraint_type == 'all':
                min_size, max_size = sizes.min(), sizes.max()
                max_scale = self.bbox_size_range[1] / max_size
                min_scale = self.bbox_size_range[0] / min_size
            elif constraint_type == 'mean':
                size = sizes.mean()
                max_scale = self.bbox_size_range[1] / size
                min_scale = self.bbox_size_range[0] / size
            elif constraint_type == 'one':
                min_size, max_size = sizes.min(), sizes.max()
                max_scale = self.bbox_size_range[1] / min_size
                min_scale = self.bbox_size_range[0] / max_size
            else:
                raise ValueError("constraint_type '{}' is unknown, must be one of ['all', 'mean', 'one']"
                                 .format(constraint_type))
            if min_scale >= max_scale:
                logger("BG2, no scale in {} can scale selected boxes to {}".format(
                    self.bbox_size_range, (min_scale, max_scale)), 'DEBUG')

            min_scale = max(min_scale, self.scale_range[0])
            max_scale = min(max_scale, self.scale_range[1])
            if min_scale >= max_scale:
                logger("scale_constraint_type={}, BG2, scale range {} is empty.".format(
                    constraint_type, (min_scale, max_scale)), 'DEBUG')
                # warnings.warn('RandomPadScaleForBBox failed, min_scale{} >= max_scale{}, cause bbox size({}, {})'
                #              ' variances are too big.'.format(min_scale, max_scale, min_size, max_size))
                return None
        scale = np.random.random() * (max_scale - min_scale) + min_scale
        # print('sizes', min_size, max_size, min_scale, max_scale, scale)
        return scale

    def choose_crop(self, crop_w_before_scale, crop_h_before_scale, image_width, image_height,
                    scale, boxes: np.ndarray, constraint_type='all'):
        """
        :param crop_w_before_scale:
        :param crop_h_before_scale:
        :param scale:
        :param boxes:
        :param constraint_type: option in ['all', 'one']
                                'all' means crop must cover all boxes
                                'one' means crop cover at least one of boxes
        :return:
        """
        if self.info_collector is not None:
            self._analysis_info['function_call']['crop_constraint_type=="{}"'.format(constraint_type)] += 1
        crop_w_after_scale, crop_h_after_scale = crop_w_before_scale / scale, crop_h_before_scale / scale
        # crop can out of origin image, but must contain >=1 gt boxes.
        if constraint_type == 'all':
            cover_box = [np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])]
        elif constraint_type == 'one':
            cover_box = boxes[np.random.randint(len(boxes))]
        else:
            raise ValueError("constrain_type '{}' is unknown, must be one of ['all', 'one']".format(constraint_type))
        min_image_x1, min_image_y1 = cover_box[2] - crop_w_after_scale, cover_box[3] - crop_h_after_scale
        max_image_x1, max_image_y1 = cover_box[0], cover_box[1]
        if min_image_x1 >= max_image_x1 or min_image_y1 >= max_image_y1:
            logger('crop_constraint_type={}, BG3, no crop box (w={},h={}) can cover selected boxes {}'.format(
                constraint_type, crop_w_after_scale, crop_h_after_scale, cover_box), 'DEBUG')
            return None

        # TODO: must change to insert of image(not union) as crop area ratio
        # crop area must cover min_crop_size_ratio^2 of origin image, to avoid too least foreground info.
        if self.min_crop_size_ratio > 0:
            min_w, min_h = crop_w_after_scale * self.min_crop_size_ratio, crop_h_after_scale * self.min_crop_size_ratio
        else:
            min_w, min_h = np.array(self.min_crop_overlap) / scale
        if crop_w_after_scale < image_width:
            min_image_x1 = max(min_image_x1, min_w - crop_w_after_scale)
            max_image_x1 = min(max_image_x1, image_width - min_w)
        else:  # contain all image if crop bigger than all image
            min_image_x1 = max(min_image_x1, image_width - crop_w_after_scale)
            max_image_x1 = min(max_image_x1, 0)
        if crop_h_after_scale < image_height:
            min_image_y1 = max(min_image_y1, min_h - crop_h_after_scale)
            max_image_y1 = min(max_image_y1, image_height - min_h)
        else:  # contain all image if crop bigger than all image
            min_image_y1 = max(min_image_y1, image_height - crop_h_after_scale)
            max_image_y1 = min(max_image_y1, 0)
        if min_image_x1 >= max_image_x1 or min_image_y1 >= max_image_y1:
            logger('crop_overlap_constraint, BG3, no crop box can have >= {}% in origin image'.format(
                self.min_crop_size_ratio * 100), 'DEBUG')
            # warnings.warn('RandomPadScaleForBBox failed, no crop available can find for scale.')
            return None

        # random choose a crop available
        image_x1 = floor(np.random.random() * (max_image_x1 - min_image_x1) + min_image_x1)
        image_y1 = floor(np.random.random() * (max_image_y1 - min_image_y1) + min_image_y1)
        image_x2 = ceil(image_x1 + crop_w_after_scale)
        image_y2 = ceil(image_y1 + crop_h_after_scale)
        crop = (image_x1, image_y1, image_x2, image_y2)
        # print(boxes, min_image_x1, max_image_x1, min_image_y1, max_image_y1)
        return crop

    def crop_with_no_scale_constraint(self, image, target):
        boxes = target.bbox.cpu().numpy()
        W, H = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.all([W <= self.crop_size_before_scale[0], H <= self.crop_size_before_scale[1]], axis=0)]
        if len(boxes) == 0:
            logger("BG, crop {} can not smaller than all gt boxes {}".format(
                self.crop_size_before_scale, target.bbox.cpu().numpy()), 'DEBUG')
            return None

        if self.info_collector is not None:
            self._analysis_info['function_call']['crop_no_scale_constraint'] += 1

        bbox = self.choose_boxes(boxes, 1)[0]
        w, h = self.crop_size_before_scale
        min_x1, min_y1 = bbox[2] - w, bbox[3] - h
        max_x1, max_y1 = bbox[0], bbox[1]
        x = np.random.randint(min_x1, max_x1+1)
        y = np.random.randint(min_y1, max_y1+1)
        crop = (x, y, x + w, y + h)
        target = self.crop_bbox(target, crop)
        image = self.crop_image(image, crop)
        return image, target


class RandomCropResizeForBBox3(RandomCropResizeForBBox2):
    """
        should combine with ScaleResize(scale)
       constrain:
           bbox constrain : at least have a gt box;
           scale constrain: 1. gt box scale to wanted [bbox_size_range]
                            2. cover at least [min_crop_size_ratio]^2 of origin image
                            3. scale must between [scale_range]
           translate      : cover at least [min_crop_size_ratio]^2 of origin image
       Method1:
           1. try getting a scale s that keep all gt boxes in origin image in [bbox_size_range] and s in [scale_range]
               -> choose_scale
           2. get crop's width and height (new_image_w, new_image_h) to origin image r.s.t scale s.
           3. try getting a crop that cover one of (leftest, topest, rightest, most bottom) gt box at least,
               and crop's left up point must left and up to center of origin image.  -> choose_crop
       Method2:
           1. random choose one of gt boxes B
           2. get a scale s to keep B size in [bbox_size_range] and s in [scale_range], get crop's width and height
           3. get a crop start_point that cover B and cover [min_crop_size_ratio]^2 of origin image at least
       """

    def __init__(self, bbox_size_range, max_crop_wh, fill=0, scale_range=(0, inf),
                 min_crop_size_ratio=0.5, min_crop_overlap=None, scale_constraint_type='all', crop_constrain_type='all',
                 constraint_auto=False, transform_prob=1., translate_range=(-16, 16), info_collector=None):
        super().__init__(bbox_size_range, max_crop_wh, fill, scale_range, min_crop_size_ratio, min_crop_overlap,
                         scale_constraint_type, crop_constrain_type, constraint_auto, transform_prob, info_collector)
        self.max_crop_size = max_crop_wh
        self.translate_range = translate_range
        self.set_random_seed = False

    def __call__(self, image: Image.Image, target: BoxList):
        """
        1. image gt box size is (s1 <= size <=s2), we choose a scale [s] r.s.t uniform(bbox_size_range[0]/s1, bbox_size_range[1]/s2)
           to make sure all gt box's size in image is: bbox_size_range[0] <= s * s1 <= s * s2 <= bbox_size_range[1].
        2. cal new_image's width and height respect tp scale [s].
        3. set origin image set axi(left-top is (0, 0)), choose a crop(new_image_x1, new_image_y1, new_image_x2, new_image_y2)
            respect ro new_image's width and height and include a gt box at least.
        4. move and crop annotation
        :param image:
        :param target:
        :return:
        """
        if self.info_collector is not None:
            self._analysis_info = {'function_call':
                                       {'deal_background': 0, 'crop_constraint_type=="all"': 0,
                                        'crop_constraint_type=="one"': 0,
                                        'scale_constraint_type=="all"': 0, 'scale_constraint_type=="one"': 0,
                                        'scale_constraint_type=="mean"': 0, 'success': 0,
                                        'crop_no_scale_constraint': 0},
                                   'statistic':
                                       {'count(<max_size)': 0}
                                   }
        if not self.set_random_seed:
            seed = int(time.time() + os.getpid())
            np.random.seed(seed)
            self.set_random_seed = True

        # print('image w, h', image.width, image.height)
        if np.random.uniform(0, 1) > self.transform_prob:  # whether use expand
            return self.deal_background(image, target)
        old_image, old_target = copy.deepcopy(image), copy.deepcopy(target)
        if True:
            # try:
            # 1. filter ignore and too big gt out, just to choose a scale and crop, final return will keep it.
            # TODO: should be replace with other policy to remove most ignore.
            boxes = target.bbox.cpu().numpy()
            # non_ignore_boxes = np.all([boxes[:, 2] - boxes[:, 0] < self.MAX_GT_WH,
            #                            boxes[:, 3] - boxes[:, 1] < self.MAX_GT_WH], axis=0)
            # boxes = boxes[non_ignore_boxes]
            # if len(boxes) == 0:
            #     logger("BG1, no non-ignore gt boxes found, boxes is {}".format(target.bbox.cpu().numpy()), 'DEBUG')
            #     return self.deal_background(old_image, old_target)

            # 2. choose boxes as constraint
            boxes = self.choose_boxes(boxes)

            # 3. choose a scale and a crop r.s.t the scale
            if self.constraint_auto:  # constraint_auto will try ['all', 'mean', 'one'] one by one.
                for constraint_type in ['all', 'mean', 'one']:
                    scale = self.choose_scale(boxes, constraint_type)
                    if scale is not None:
                        break
            else:
                scale = self.choose_scale(boxes, self.scale_constraint_type)

            if scale is None:
                return self.deal_background(old_image, old_target)

            # 4. choose a crop
            if self.crop_size_before_scale is None:
                crop_w_before_scale, crop_h_before_scale = image.width, image.height
            else:
                crop_w_before_scale, crop_h_before_scale = self.crop_size_before_scale

            if self.constraint_auto:
                for constrain_type in ['all', 'one']:
                    crop = self.choose_crop(crop_w_before_scale, crop_h_before_scale, image.width, image.height,
                                            scale, boxes, constrain_type)
                    if crop is not None:
                        break
            else:
                crop = self.choose_crop(crop_w_before_scale, crop_h_before_scale, image.width, image.height,
                                        scale, boxes, self.crop_constrain_type)  # crop can out of origin image

            if crop is None:
                return self.deal_background(old_image, old_target)
            # print(scale, crop, crop[2]-crop[0], crop[3]-crop[1])

            # 5. crop bbox and image r.s.t choose crop
            # if scale < 1, we can scale image fist and then crop to speed up
            image = self.crop_image(image, crop, scale if scale < 1 else None)
            target = self.crop_bbox(target, crop, image.size)

            # 6. scale image and bbox
            need_scale = scale > 1
            image_size = (np.array(image.size) * scale).astype(np.int32)
            if image_size[0] > self.max_crop_size[0]:  # for int get bigger input, like max_crop_size + 1
                image_size[0] = self.max_crop_size[0]
                need_scale = True
            if image_size[1] > self.max_crop_size[1]:
                image_size[1] = self.max_crop_size[1]
                need_scale = True
            if need_scale:
                image = image.resize(image_size)
                target.resize(image.size)

        # except BaseException as e:
        #     # print(e)
        #     # warnings.warn("exception happened which should not happened, may be some bug in code.")
        #     # raise e
        #     return self.deal_background(old_image, old_target)
        # print(crop[2]-crop[0], crop[3]-crop[1], image.size)

        if self.info_collector is not None:
            self._analysis_info['function_call']['success'] += 1
            self.info_collector(self._analysis_info)
        return image, target

    def choose_crop(self, crop_max_w, crop_max_h, image_width, image_height,
                    scale, boxes: np.ndarray, constraint_type='all'):
        """
        :param crop_max_w:
        :param crop_max_h:
        :param scale:
        :param boxes:
        :param constraint_type: option in ['all', 'one']
                                'all' means crop must cover all boxes
                                'one' means crop cover at least one of boxes
        :return:
        """
        if self.info_collector is not None:
            self._analysis_info['function_call']['crop_constraint_type=="{}"'.format(constraint_type)] += 1
        crop_w_after_scale, crop_h_after_scale = crop_max_w / scale, crop_max_h / scale
        # crop can out of origin image, but must contain >=1 gt boxes.
        if constraint_type == 'all':
            cover_box = [np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])]
        elif constraint_type == 'one':
            cover_box = boxes[np.random.randint(len(boxes))]
        else:
            raise ValueError("constrain_type '{}' is unknown, must be one of ['all', 'one']".format(constraint_type))
        min_image_x1, min_image_y1 = cover_box[2] - crop_w_after_scale, cover_box[3] - crop_h_after_scale
        max_image_x1, max_image_y1 = cover_box[0], cover_box[1]
        if min_image_x1 >= max_image_x1 or min_image_y1 >= max_image_y1:
            logger('crop_constraint_type={}, BG3, no crop box (w={},h={}) can cover selected boxes {}'.format(
                constraint_type, crop_w_after_scale, crop_h_after_scale, cover_box), 'DEBUG')
            return None

        # TODO: must change to insert of image(not union) as crop area ratio
        # crop area must cover min_crop_size_ratio^2 of origin image, to avoid too least foreground info.
        if self.min_crop_size_ratio > 0:
            min_w, min_h = crop_w_after_scale * self.min_crop_size_ratio, crop_h_after_scale * self.min_crop_size_ratio
        else:
            min_w, min_h = np.array(self.min_crop_overlap) / scale
        if crop_w_after_scale < image_width:
            min_image_x1 = max(min_image_x1, min_w - crop_w_after_scale)
            max_image_x1 = min(max_image_x1, image_width - min_w)
        else:  # contain all image if crop bigger than all image
            min_image_x1, max_image_x1 = self.translate_range
            crop_w_after_scale = image_width
            # self._analysis_info['statistic']['count(<max_size)'] += 1
        if crop_h_after_scale < image_height:
            min_image_y1 = max(min_image_y1, min_h - crop_h_after_scale)
            max_image_y1 = min(max_image_y1, image_height - min_h)
        else:  # contain all image if crop bigger than all image
            min_image_y1, max_image_y1 = self.translate_range
            crop_h_after_scale = image_height
            # self._analysis_info['statistic']['count(<max_size)'] += 1
        if min_image_x1 >= max_image_x1 or min_image_y1 >= max_image_y1:
            logger('crop_overlap_constraint, BG3, no crop box can have >= {}% in origin image'.format(
                self.min_crop_size_ratio * 100), 'DEBUG')
            # warnings.warn('RandomPadScaleForBBox failed, no crop available can find for scale.')
            return None

        # random choose a crop available
        image_x1 = floor(np.random.random() * (max_image_x1 - min_image_x1) + min_image_x1)
        image_x2 = ceil(image_x1 + crop_w_after_scale)
        image_y1 = floor(np.random.random() * (max_image_y1 - min_image_y1) + min_image_y1)
        image_y2 = ceil(image_y1 + crop_h_after_scale)
        crop = (image_x1, image_y1, image_x2, image_y2)
        # print(boxes, min_image_x1, max_image_x1, min_image_y1, max_image_y1)
        # print('crop', np.array(crop) * scale, image_height, image_width, crop_h_after_scale, crop_w_after_scale, scale)
        return crop

    def crop_with_no_scale_constraint(self, image, target):
        boxes = target.bbox.cpu().numpy()
        W, H = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.all([W <= self.crop_size_before_scale[0], H <= self.crop_size_before_scale[1]], axis=0)]
        if len(boxes) == 0:
            logger("BG, crop {} can not smaller than all gt boxes {}".format(
                self.crop_size_before_scale, target.bbox.cpu().numpy()), 'DEBUG')
            return None

        if self.info_collector is not None:
            self._analysis_info['function_call']['crop_no_scale_constraint'] += 1

        bbox = self.choose_boxes(boxes, 1)[0]
        w, h = self.crop_size_before_scale
        min_x1, min_y1 = bbox[2] - w, bbox[3] - h
        max_x1, max_y1 = bbox[0], bbox[1]
        x = np.random.randint(min_x1, max_x1+1)
        y = np.random.randint(min_y1, max_y1+1)
        crop = (x, y, x + w, y + h)
        target = self.crop_bbox(target, crop)
        image = self.crop_image(image, crop)
        return image, target

# class RandomCropWithConstraints(object):
#     def __init__(self, min_scale=0.3, max_scale=1, max_aspect_ratio=2, constraints=None, max_trial=50,
#                  transform_prob=1.0):
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#         self.max_aspect_ratio = max_aspect_ratio
#         self.constraints = constraints
#         self.max_trial = max_trial
#         self.transform_prob = transform_prob
#
#     def __call__(self, image: Image.Image, target: BoxList):
#         if np.random.uniform(0, 1) > self.transform_prob:                   # whether use expand
#             return image, target
#         bbox, crop = self.random_crop_with_constraints(target.convert('xyxy').bbox, target.size)
#         left, top, right, bottom = crop
#         target = BoxList(bbox, (right-left, bottom-top), 'xyxy').convert(target.mode)
#         image = image.crop(crop)
#         return image, target
#
#     def random_crop_with_constraints(self, bbox, size, min_scale=0.3, max_scale=1,
#                                      max_aspect_ratio=2, constraints=None,
#                                      max_trial=50):
#         """Crop an image randomly with bounding box constraints.
#
#         This data augmentation is used in training of
#         Single Shot Multibox Detector [#]_. More details can be found in
#         data augmentation section of the original paper.
#         .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
#            Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
#            SSD: Single Shot MultiBox Detector. ECCV 2016.
#
#         Parameters
#         ----------
#         bbox : numpy.ndarray
#             Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
#             The second axis represents attributes of the bounding box.
#             Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
#             we allow additional attributes other than coordinates, which stay intact
#             during bounding box transformations.
#         size : tuple
#             Tuple of length 2 of image shape as (width, height).
#         min_scale : float
#             The minimum ratio between a cropped region and the original image.
#             The default value is :obj:`0.3`.
#         max_scale : float
#             The maximum ratio between a cropped region and the original image.
#             The default value is :obj:`1`.
#         max_aspect_ratio : float
#             The maximum aspect ratio of cropped region.
#             The default value is :obj:`2`.
#         constraints : iterable of tuples
#             An iterable of constraints.
#             Each constraint should be :obj:`(min_iou, max_iou)` format.
#             If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
#             If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
#             (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
#         max_trial : int
#             Maximum number of trials for each constraint before exit no matter what.
#
#         Returns
#         -------
#         numpy.ndarray
#             Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
#         tuple
#             Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
#
#         """
#         # default params in paper
#         if constraints is None:
#             constraints = (
#                 (0.1, None),
#                 (0.3, None),
#                 (0.5, None),
#                 (0.7, None),
#                 (0.9, None),
#                 (None, 1),
#             )
#
#         if len(bbox) == 0:
#             constraints = []
#
#         w, h = size
#
#         candidates = [(0, 0, w, h)]
#         for min_iou, max_iou in constraints:
#             min_iou = -np.inf if min_iou is None else min_iou
#             max_iou = np.inf if max_iou is None else max_iou
#
#             for _ in range(max_trial):
#                 scale = random.uniform(min_scale, max_scale)
#                 aspect_ratio = random.uniform(
#                     max(1 / max_aspect_ratio, scale * scale),
#                     min(max_aspect_ratio, 1 / (scale * scale)))
#                 crop_h = int(h * scale / np.sqrt(aspect_ratio))
#                 crop_w = int(w * scale * np.sqrt(aspect_ratio))
#
#                 crop_t = random.randrange(h - crop_h)
#                 crop_l = random.randrange(w - crop_w)
#                 crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))
#
#                 iou = boxlist_iou(BoxList(bbox, size), BoxList(crop_bb[np.newaxis], size))
#                 if min_iou <= iou.min() and iou.max() <= max_iou:
#                     top, bottom = crop_t, crop_t + crop_h
#                     left, right = crop_l, crop_l + crop_w
#                     candidates.append((left, top, right, bottom))
#                     break
#
#         # random select one
#         while candidates:
#             crop = candidates.pop(np.random.randint(0, len(candidates)))
#             left, top, right, bottom = crop
#             bboxlist = BoxList(bbox, size).translate(-left, -top)
#             new_bbox = bboxlist.crop((0, 0, right-left, bottom-top)).clip_to_image().bbox
#             if len(new_bbox) < 1:
#                 continue
#             print(new_bbox)
#             new_crop = (crop[0], crop[1], crop[2], crop[3])
#             return new_bbox, new_crop
#         return bbox, (0, 0, w, h)


####################################################################################
