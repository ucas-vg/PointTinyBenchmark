# TinyCOCO

## 1. annotation
### 1.1 point annotation prepare

rename 'bbox' to 'true_bbox', and add 'point'
```sh
export VERSION=1
export CORNER=""
# export T="val"
export T="train"
python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/coco/annotations/instances_${T}2017.json" \
    "data/coco/coarse_annotations/noise_uniform_${VERSION}/${CORNER}/instances_${T}2017_coarse.json"

export MU=(0 0)
export S=(0.25 0.25)  # sigma
# export S=(0.167 0.167)  # sigma
# export S=(0.125 0.125)  # sigma
export VERSION=1
export CORNER=""
# export T="val"
export T="train"
python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/coco/annotations/instances_${T}2017.json" \
    "data/coco/coarse_annotations/noise_rg-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}_${VERSION}/${CORNER}/instances_${T}2017_coarse.json" \
    --rand_type 'range_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})"

> 860001it [45:48, 312.91it/s]

# show point distrubtution
python exp/tools/plot_gaussian.py
```

### 1.2 generate pseudo box

add 'bbox' and remove 'segmentation'
```
export VERSION=1
export CORNER=""
export WH=(16 16)
export T="train"
python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/LVIS/coarse_annotations/noise_uniform_${VERSION}/${CORNER}/instances_train2017_coarse.json" \
    "data/LVIS/coarse_gen_annotations/noise_uniform_${VERSION}/${CORNER}/pseuw${WH[0]}h${WH[1]}/instances_train2017_coarse.json" \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}

export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export RS=0.25
export CORNER=""
export WH=(64 64)
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/LVIS/coarse_annotations/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${RS}_${VERSION}/lvis_v1_train_coarse.json" \
    "data/LVIS/coarse_gen_annotations/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${RS}_${VERSION}/${CORNER}/pseuw${WH[0]}h${WH[1]}/lvis_v1_train_coarse.json" \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}

> annotations count from 860001 to 860001
> save pseudo bbox annotation file in data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1//pseuw32h32/instances_train2017_coarse.json
```


### quasi-center point annotation

```sh
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25 # size_range
export VERSION=1
export CORNER=""
# export T="val"
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/coco/annotations/instances_${T}2017.json" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/instances_${T}2017_coarse.json" \
    --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
    --size_range "${SR}"
```

# voc quasi-center point annotation
```sh
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25 # size_range
export VERSION=1
export CORNER=""
# export T="val"
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/VOC2012_SBD/cocostyle/voc12sbd_ins_${T}_cls.json" \
    "data/VOC2012_SBD/cocostyle_coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_voc12sbd_ins_${T}_cls.json.json" \
    --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
    --size_range "${SR}"
```

# cityscape qc point annotation

```sh
export VERSION=1
export CORNER=""
export WH=(16 16)
export T="train"
python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/LVIS/coarse_annotations/noise_uniform_${VERSION}/${CORNER}/instances_train2017_coarse.json" \
    "data/LVIS/coarse_gen_annotations/noise_uniform_${VERSION}/${CORNER}/pseuw${WH[0]}h${WH[1]}/instances_train2017_coarse.json" \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}

export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export RS=0.25
export CORNER=""
export WH=(64 64)
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "/home/ubuntu/disk2/dataset/cityscapes/qc_instancesonly_filtered_gtFine_train.json"  \
    "/home/ubuntu/disk2/dataset/cityscapes/qc_instancesonly_filtered_gtFine_train_with_bbox.json"  \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}
```