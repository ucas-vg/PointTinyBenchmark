# TinyPersonV2

```
ln -s $PATH_TO_TinyPersonV2$ data/tiny_set_v2
```

### 1. generate corner dataset
```shell script
export CORNER=(640 640)
export OVERLAP=(100 100)
python huicv/corner_dataset/corner_dataset_util.py data/tiny_set_v2/anns/release/rgb_train.json \
    data/tiny_set_v2/anns/release/corner/rgb_train_w${CORNER[0]}h${CORNER[1]}ow${OVERLAP[0]}oh${OVERLAP[1]}.json \
    --max_tile_w ${CORNER[0]} --max_tile_h ${CORNER[1]} --tile_overlap_w ${OVERLAP[0]} --tile_overlap_h ${OVERLAP[1]} \
    --ann_type 'bbox'
    
export CORNER=(640 640)
export OVERLAP=(100 100)
python huicv/corner_dataset/corner_dataset_util.py data/tiny_set_v2/anns/release/rgb_valid.json \
    data/tiny_set_v2/anns/release/corner/rgb_valid_w${CORNER[0]}h${CORNER[1]}ow${OVERLAP[0]}oh${OVERLAP[1]}.json \
    --max_tile_w ${CORNER[0]} --max_tile_h ${CORNER[1]} --tile_overlap_w ${OVERLAP[0]} --tile_overlap_h ${OVERLAP[1]} \
    --ann_type 'bbox'

export CORNER=(640 640)
export OVERLAP=(100 100)
python huicv/corner_dataset/corner_dataset_util.py data/tiny_set_v2/anns/release/rgb_trainvalid.json \
    data/tiny_set_v2/anns/release/corner/rgb_trainvalid_w${CORNER[0]}h${CORNER[1]}ow${OVERLAP[0]}oh${OVERLAP[1]}.json \
    --max_tile_w ${CORNER[0]} --max_tile_h ${CORNER[1]} --tile_overlap_w ${OVERLAP[0]} --tile_overlap_h ${OVERLAP[1]} \
    --ann_type 'bbox'

export CORNER=(640 640)
export OVERLAP=(100 100)
python huicv/corner_dataset/corner_dataset_util.py data/tiny_set_v2/anns/release/rgb_test.json \
    data/tiny_set_v2/anns/release/corner/rgb_test_w${CORNER[0]}h${CORNER[1]}ow${OVERLAP[0]}oh${OVERLAP[1]}.json \
    --max_tile_w ${CORNER[0]} --max_tile_h ${CORNER[1]} --tile_overlap_w ${OVERLAP[0]} --tile_overlap_h ${OVERLAP[1]} \
    --ann_type 'bbox'
```

### 2. dataset config

```shell script
cp configs2/_base_/datasets/TinyPerson/TinyPerson_detection_640x640.py configs2/_base_/datasets/TinyPersonV2/TinyPersonV2_detection_640x640.py
# modified annotation and image path: data.train, data.val, data.test
```

### 3. model config

```shell script
cp configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py configs2/TinyPersonV2/base/faster_rcnn_r50_fpn_1x_TinyPersonV2_640.py
```

### 4. run experiment

```shell script
# exp1.1: Faster-FPN, 2GPU
export GPU=2 && LR=0.01 && B=2 && WH=(640 640) && CONFIG="TinyPersonV2/base/faster_rcnn_r50_fpn_1x_TinyPersonV2_640" && \
CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/${CONFIG}/trainval${WH[0]}x${WH[1]}_lr${LR}_1x_b${B}${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B} \
    data.train.ann_file="data/tiny_set_v2/anns/release/corner/rgb_trainvalid_w${WH[0]}h${WH[1]}ow100oh100.json" \
    data.val.ann_file="data/tiny_set_v2/anns/release/rgb_test.json"

# exp1.1: Faster-FPN, 2GPU
export GPU=2 && LR=0.01 && B=2 && WH=(640 640) && CONFIG="TinyPersonV2/base/faster_rcnn_r50_fpn_1x_TinyPersonV2_640" && \
CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/${CONFIG}/trainval${WH[0]}x${WH[1]}_lr${LR}_1x_b${B}${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B} \
    data.train.ann_file="data/tiny_set_v2/anns/release/corner/rgb_trainvalid_w${WH[0]}h${WH[1]}ow100oh100.json" \
    data.val.ann_file="data/tiny_set_v2/anns/release/rgb_test.json"
```

- GPU: 3080 x2
- train set: train + val; eval set: test

detector | lr | batch |$AP_{50}$ | $AP_{50}^{tiny}$ | script
--- | :---: | :---: | :---: | :---: | ---
Faster RCNN | 0.01  | 1x2 | ~~67.17~~<br>73.19 | ~~54.11~~<br>60.63 | exp/sh/Baseline_TinyPersonV2.sh:1.1
Faster RCNN | 0.02  | 2x2 | ~~67.90~~<br>73.52 | ~~54.32~~<br>60.85 | exp/sh/Baseline_TinyPersonV2.sh:1.2
Faster RCNN | 0.04  | 4x2 | ~~67.36~~<br>73.44 | ~~54.51~~<br>60.78 | exp/sh/Baseline_TinyPersonV2.sh:1.3
--- | ---| --- | --- | ---| ---
Faster RCNN | 0.02  | 4x2 | ~~66.36~~<br>71.98 | ~~53.32~~<br>59.55 | exp/sh/Baseline_TinyPersonV2.sh:1.4
Faster RCNN | 0.04  | 4x2 | ~~67.36~~<br>73.44 | ~~54.51~~<br>60.78 | exp/sh/Baseline_TinyPersonV2.sh:1.3
Faster RCNN | 0.06  | 4x2 | ~~67.99~~<br>73.56 | ~~54.56~~<br>61.06 | exp/sh/Baseline_TinyPersonV2.sh:1.5



detector | cut size | lr | batch| $AP_{50}$ | $AP_{50}^{tiny}$ | script
--- | :---: | :---: | :---: | :---: | :---: | ---
Faster RCNN | (320, 320) | 0.04  | 4x2 | ~~61.21~~<br>67.67 | ~~49.94~~<br>58.43 | exp/sh/Baseline_TinyPersonV2.sh:1.7
Faster RCNN | (640, 640) | 0.04  | 4x2 | ~~67.36~~<br>73.44 | ~~54.51~~<br>60.78 | exp/sh/Baseline_TinyPersonV2.sh:1.3
Faster RCNN | (960, 960) | 0.04  | 4x2 | ~~61.23~~<br>[-] | ~~50.02~~<br>[-] | exp/sh/Baseline_TinyPersonV2.sh:1.6.1
Faster RCNN | (960, 960) | 0.04  | 2x2 | ~~67.33~~<br>72.77 | ~~52.89~~<br>59.53 | exp/sh/Baseline_TinyPersonV2.sh:1.6.2

#### big table

- cut size: (640, 640)

detector | batch | lr | $AP_{50}$ | $AP_{50}^{tiny}$ | script
--- | :---: | :---: | :---: | :---: | ---
Faster RCNN     | 4x2 | 0.04 | ~~67.36~~<br>73.44 | ~~54.51~~<br>60.78 | exp/sh/Baseline_TinyPersonV2.sh:1.3
RetinaNet       | 4x2 | 0.04 | ~~54.03~~<br>72.75 | ~~42.11~~<br>59.45 | exp/sh/Baseline_TinyPersonV2.sh:2.1
Adap RetinaNet  | 3x2 | 0.04 | 74.19 | 63.78 | exp/sh/Baseline_TinyPersonV2.sh:2.2
FCOS            | 4x2 | 0.04 | ~~48.09~~<br>65.08 | ~~41.39~~<br>54,37 | exp/sh/Baseline_TinyPersonV2.sh:3.1
Adap FCOS       | 4x2 | 0.04 | ~~65.32~~<br>71.59 | ~~54.02~~<br>60.98 | exp/sh/Baseline_TinyPersonV2.sh:3.2
RepPoint        | 4x2 | 0.04 | ~~54.71~~<br>73.47 | ~~43.18~~<br>60.79 | exp/sh/Baseline_TinyPersonV2.sh:4.1
Adap RepPoint   | 4x2 | 0.04 | ~~68.92~~<br>75.27 | ~~56.17~~<br>64.56 | exp/sh/Baseline_TinyPersonV2.sh:4.2

- 

detector | DA | batch | lr | num_proposal | NMS| $AP_{50}$ | $AP_{50}^{tiny}$ | script
--- | ---| --- | --- | --- | --- | --- | ---|---
Sparse RCNN | N               | 1x2 | 2.5e-5 | 500 | N | | 
Sparse RCNN | N               | 1x2 | 2.5e-5 | 1000 | N | | 
Sparse RCNN | MS train        | 1x2 | 2.5e-5 | 1000 |
Sparse RCNN | MS train + crop | 1x2 | 2.5e-5 | 1000 |

#### batch size

 detector | batch x gpu | lr | $AP_{50}$ | $AP_{50}^{tiny}$ | script
 Adap RepPoint | 4x4 | 0.08 | |
 
 