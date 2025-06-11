import json

# ori_file = 'data/coco/coarse_gen_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/my_instances_train2014_partial.json'
# src_file = 'data/coco/coarse_gen_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/instances_train2014_coarse.json'
# dst_file = 'data/coco/coarse_gen_annotations/quasi-center-point-0-0-0.25-0.25-0.3_1/instances_train2014_coarse_with_seg.json'
ori_file='/home/pfchen/disk1/cpf/P2BNet/TOV_mmdetection/data/VOC2007/annotations/voc_2007_trainval.json'
src_file='/home/pfchen/disk1/cpf/P2BNet/TOV_mmdetection/data/VOC2007/Annotations-QC-0-0-0.25-0.25-0.25_coco_fmt/voc07_trainval.json'
dst_file='/home/pfchen/disk1/cpf/P2BNet/TOV_mmdetection/data/VOC2007/Annotations-QC-0-0-0.25-0.25-0.25_coco_fmt/voc07_trainval_with_seg.json'
ori_jd = json.load(open(ori_file, 'r'))
src_jd = json.load(open(src_file, 'r'))
# dst_jd = json.load(open(dst_file, 'r'))
ann = src_jd['annotations']
ann_ori = ori_jd['annotations']
for i in range(len(ann)):
    print(i)
    ann[i]['segmentation'] = ann_ori[i]['segmentation']

json.dump(src_jd, open(dst_file, 'w'))
print('a')