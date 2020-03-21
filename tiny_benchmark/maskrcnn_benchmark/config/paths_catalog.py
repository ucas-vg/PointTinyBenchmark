# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/$user/TinyBenchmark/dataset"
    DATASETS = {
        "coco_2017_merge": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/instances_merge2017.json"
        },
        "coco_2017_merge_sample4": {
            "img_dir": "coco/images",
            "ann_file": "coco/annotations/instances_merge2017_sample4.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/images",  # "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_merge": {
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014.json"
        },
        "coco_2014_merge_only_person_anno": {
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014_only_person_anno.json"
        },
        "coco_2014_merge_no_person_anno": {
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014_no_person_anno.json"
        },
        "coco_2014_merge_no_person_image": {
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014_no_person_image.json"
        },
        "coco_2014_merge_clipAspectCity": {
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014_clipAspectCity.json"
        },
        "coco_2014_merge_sample4":{
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_simple_merge2014_sample4.json"
        },
        "coco_2014_apect2.44_merge":{
            "img_dir": "coco/",
            "ann_file": "coco/annotations/instances_merge2014_aspect2.44.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_minival_resize": {  # can not use as train
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014_resize.json"  # only area resized to eval (FCOS train use area)
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },

        # voc debug #########################################################
        "voc_2007_train_sub0_2_debug_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007_sub0_2.json"
        },
        "voc_2007_train_sub0_32_debug_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007_sub0_32.json"
        },
        # voc debug #########################################################

        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test",
            'ann_file': "voc/VOC2007/Annotations"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },

        # cityperson pedestrian
        "cityperson_pedestrian_train_coco": {
            "img_dir": "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",
            "ann_file": "cityscapes/perdestrian_annotations/citypersons_all_train.json"
        },
        "cityperson_pedestrian_erase_train_coco": {
            "img_dir": "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train_erase",
            "ann_file": "cityscapes/perdestrian_annotations/citypersons_all_train_erase.json"
        },
        "cityperson_pedestrian_val_coco": {
            "img_dir": "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val",
            "ann_file": "cityscapes/perdestrian_annotations/citypersons_all_val.json"
        },
        "cityperson_pedestrian_erase_train_corner_sw576_sh576_coco": {
            "img_dir": "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train_erase",
            "ann_file": "cityscapes/perdestrian_annotations/corner/citypersons_all_train_erase_sw576_sh576.json"
        },
        "cityperson_pedestrian_val_corner_sw576_sh576_coco": {
            "img_dir": "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val",
            "ann_file": "cityscapes/perdestrian_annotations/corner/citypersons_all_val_sw576_sh576.json"
        },

        # cityperson pedestrian
        "tiny_cityperson_pedestrian_train_coco": {
            "img_dir": "cityscapes/tiny/leftImg8bit_trainvaltest/leftImg8bit/train",
            "ann_file": "cityscapes/tiny/perdestrian_annotations/citypersons_all_train.json"
        },
        "tiny_cityperson_pedestrian_erase_train_coco": {
            "img_dir": "cityscapes/tiny/leftImg8bit_trainvaltest/leftImg8bit/train_erase",
            "ann_file": "cityscapes/tiny/perdestrian_annotations/citypersons_all_train_erase.json"
        },
        "tiny_cityperson_pedestrian_val_coco": {
            "img_dir": "cityscapes/tiny/leftImg8bit_trainvaltest/leftImg8bit/val",
            "ann_file": "cityscapes/tiny/perdestrian_annotations/citypersons_all_val.json"
        },

        # tinyperson
        "tiny_set_corner_sw640_sh512_erase_with_uncertain_train_all_coco": {
            'img_dir': 'tiny_set/erase_with_uncertain_dataset/train',
            'ann_file': 'tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json'
        },
            # origin image
        "tiny_set_corner_sw640_sh512_train_all_coco": {
            'img_dir': 'tiny_set/train',
            'ann_file': 'tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json'
        },
            # big cut
        "tiny_set_corner_sw1920_sh1080_erase_with_uncertain_train_all_coco": {
            'img_dir': 'tiny_set/erase_with_uncertain_dataset/train',
            'ann_file': 'tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw1920_sh1080_all.json'
        },
            # origin image + big cut
        "tiny_set_corner_sw1920_sh1080_train_all_coco": {
            'img_dir': 'tiny_set/train',
            'ann_file': 'tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw1920_sh1080_all.json'
        },
            # big cut
        "tiny_set_corner_sw1920_sh1080_test_all_coco": {
            'img_dir': 'tiny_set/test',
            'ann_file': 'tiny_set/annotations/corner/task/tiny_set_test_sw1920_sh1080_all.json'
        },
        "tiny_set_corner_sw640_sh512_test_all_coco": {
            'img_dir': 'tiny_set/test',
            'ann_file': 'tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json'
        },
        
        # sayna
        "sanya_all_rgb_train_pw4_ph2_cocostyle":{
            "img_dir": "sanya/cocostyle_release/all/rgb/images",
            "ann_file": "sanya/cocostyle_release/all/rgb/corner_annotations/tiny_all_rgb_train_coco_pw4_ph2.json"
        },
        "sanya_all_rgb_test_pw4_ph2_cocostyle":{
            "img_dir": "sanya/cocostyle_release/all/rgb/images",
            "ann_file": "sanya/cocostyle_release/all/rgb/corner_annotations/tiny_all_rgb_test_coco_pw4_ph2.json"
        },
        "sanya_all_rgb_minus17test_pw4_ph2_cocostyle":{
            "img_dir": "sanya/cocostyle_release/all/rgb/images",
            "ann_file": "sanya/cocostyle_release/all/rgb/corner_annotations/tiny_all_rgb_minus17test_coco_pw4_ph2.json"
        },
        "sanya_all_rgb_minus17test_cocostyle":{
            "img_dir": "sanya/cocostyle_release/all/rgb/images",
            "ann_file": "sanya/cocostyle_release/all/rgb/annotations/tiny_all_rgb_minus17test_coco.json"
        },
        "sanya17test_cocostyle":{
            "img_dir": "sanya17test/img",
            "ann_file": "sanya17test/annotation_cocostyle.json"
        },
        "sanya17test_corner_cocostyle":{
            "img_dir": "sanya17test/img",
            "ann_file": "sanya17test/annotation_cocostyle_pw4_ph2.json"
        },
        "sanya17test_voc":{
            "data_dir": "sanya17test/voc_annos",
            "split": "test",
            'ann_file': "sanya17test/Annotations"
        },
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
