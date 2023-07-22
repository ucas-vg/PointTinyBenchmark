import json
from ssdcv.deps.mini_mmcv.image.geometric import rescale_size
from ssdcv.json_dataset.coco_ann_utils import GCOCO
from ssdcv.interactivate.path_utils import makedirs_if_not_exist
from PIL import Image
import numpy as np
try:
    from tqdm import tqdm
except ImportError as e:
    tqdm = lambda x: x


def resize_dataset(ann_file, image_root, out_ann_file, out_image_root, target_im_size, area_use='bbox', jpg_quality=None):
    """
        target_im_size: coco default is (800, 1333) while training and inference by default.
    """
    dataset = GCOCO(ann_file)
    for im_id in tqdm(dataset.imgs):
        im_info = dataset.imgs[im_id]
        anns = dataset.imgToAnns[im_id]
        im_size = (im_info['width'], im_info['height'])
        new_size, scale_factor = rescale_size(im_size, target_im_size, return_scale=True)

        # save new image
        if image_root is not None and out_image_root is not None:
            src_img_path = f"{image_root}/{im_info['file_name']}"
            dst_img_path = f"{out_image_root}/{im_info['file_name']}"
            img = Image.open(src_img_path)
            img = img.resize(new_size, Image.ANTIALIAS)
            makedirs_if_not_exist(dst_img_path)
            if jpg_quality is not None and (dst_img_path.endswith('.jpg') or dst_img_path.endswith('.jpeg')):
                img.save(dst_img_path, quality=jpg_quality)
            else:
                # if not (dst_img_path.endswith('.jpg') or dst_img_path.endswith('.jpeg')):
                print(f"not jpeg suffix save path {dst_img_path}")
                img.save(dst_img_path)

        # calculation of area_of_seg depend on img size., change before modified bbox
        im_info['width'] = new_size[0]
        im_info["height"] = new_size[1]
        for ann in anns:
            # use area of bbox, not segmentation
            dataset.resize_ann(ann, scale_factor, scale_factor, area_use=area_use, inplace=True, ignore_rle=True)

    dataset.save(out_ann_file, n_round=3)
    print(f"save ann on {out_ann_file}")


def parser_none_if_empty(s, type):
    return None if len(s) == 0 else type(s)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("ann")
    parser.add_argument("img_root")
    parser.add_argument("--save-ann")
    parser.add_argument("--save-img-root")
    parser.add_argument("--im-size", default="100,167", help="the target image size of short and long edge.")
    parser.add_argument("--area-of", default="bbox", help="the 'area' in annotation will set as area of"
                                                          " 'bbox' or 'segmentation'")
    parser.add_argument("--jpg-quality", default="", help="saved jpg quality, 0-100. see "
                                                          "https://jdhao.github.io/2019/07/20/pil_jpeg_image_quality/")
    args = parser.parse_args()

    target_im_size = tuple(int(e) for e in args.im_size.split(","))
    jpg_quality = parser_none_if_empty(args.jpg_quality, int)
    img_root = parser_none_if_empty(args.img_root, str)
    save_img_root = parser_none_if_empty(args.save_img_root, str)
    resize_dataset(args.ann, img_root, args.save_ann, save_img_root, target_im_size, args.area_of, jpg_quality)

    # target_im_size = (100, 133)
    #
    # size_suffix = "{}x{}".format(*target_im_size)
    #
    # ann_file = "data/coco/instances_val2017.json"
    # image_root = "data/coco/images"
    # out_ann_file = "data/coco/resize/annotations/instances_val2017_{}.json".format(size_suffix)
    # out_image_root = "data/coco/resize/images_{}".format(size_suffix)
