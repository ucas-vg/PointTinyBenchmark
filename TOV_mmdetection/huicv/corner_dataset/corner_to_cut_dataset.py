from pycocotools.coco import COCO
from PIL import Image
import json
import os

if __name__ == '__main__':
    """
    python huicv/corner_dataset/corner_to_cut_dataset.py \
        corner_ann_file img_root save_cut_ann_file save_cut_img_root
    
    python huicv/corner_dataset/corner_to_cut_dataset.py \
        data/tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json \
        data/tiny_set/erase_with_uncertain_dataset/test/ \
        data/tiny_set/mini_annotations/cut/tiny_set_test_all_erase_cut_w640h512ow100oh100.json \
        data/tiny_set/erase_with_uncertain_dataset/test_cut_w640h512ow100oh100/
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_file")
    parser.add_argument("img_root")
    parser.add_argument("save_ann")
    parser.add_argument("save_img_root")
    args = parser.parse_args()

    ann_file = args.ann_file
    img_root = args.img_root
    save_ann = args.save_ann
    save_img_root = args.save_img_root

    jd = json.load(open(ann_file))
    for img_info in jd['images']:
        file_pth = img_info['file_name']

        file_dir, file_name = os.path.split(file_pth)
        file_name, file_ext = os.path.splitext(file_name)
        corner_suffix = "_".join([str(c) for c in img_info['corner']])
        img_info['file_name'] = f"{file_dir}/{file_name}_{corner_suffix}{file_ext}"

        img_pth = os.path.join(img_root, file_pth)
        img = Image.open(img_pth)
        img = img.crop(img_info['corner'])
        del img_info['corner']

        save_img_path = os.path.join(save_img_root, img_info['file_name'])
        img.save(save_img_path)
    json.dump(jd, open(save_ann, 'w'))
