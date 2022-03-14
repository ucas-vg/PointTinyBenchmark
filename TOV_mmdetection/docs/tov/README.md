# Scale Match for TinyPerson Detection (WACV2020)

## Prerequisites
### Install enviroment
As [Install](../../README.md) said.

### prepare dataset
#### TinyPerson

to train baseline of TinyPerson, download the mini_annotation of all annotation is enough, 
which can be downloaded as tiny_set/mini_annotations.tar.gz in [Baidu Yun(password:pmcq) ](https://pan.baidu.com/s/1kkugS6y2vT4IrmEV_2wtmQ)/
[Google Driver](https://drive.google.com/open?id=1KrH9uEC9q4RdKJz-k34Q6v5hRewU5HOw).

```
mkdir data
ln -s ${Path_Of_TinyPerson} data/tiny_set
tar -zxvf data/tiny_set/mini_annotations.tar.gz && mv mini_annotations data/tiny_set/
```

dataset download link

[Official Site](http://vision.ucas.ac.cn/resource.asp): recomended, download may faster<br/>
[Baidu Pan](https://pan.baidu.com/s/1kkugS6y2vT4IrmEV_2wtmQ)   password: pmcq<br/>
[Google Driver](https://drive.google.com/open?id=1KrH9uEC9q4RdKJz-k34Q6v5hRewU5HOw)<br/>
For more details about TinyPerson dataset, please see [Dataset](https://github.com/ucas-vg/TinyBenchmark/tree/master/dataset).

#### COCO

```
ln -s ${Path_Of_COCO} data/coco
```

## Experiment
### TinyPerson

For running more experiment, to see [bash script](../../configs2/TinyPerson/base/Baseline_TinyPerson.sh)

```shell script
# exp1.2: Faster-FPN, 2GPU
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# Scale Match
# exp4.0 coco-sm-tinyperson: Faster-FPN, coco batch=8x2
export GPU=2 && export LR=0.01 && export BATCH=8 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
# python exp/tools/extract_weight.py ${COCO_WORK_DIR}/latest.pth
export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/cocosm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth
```

- GPU: 3080 x 2
- Adap RetainaNet-c means use clip grad while training.
- COCO val $mmap$ only use for debug, cause val also add to train while sm/msm coco to TinyPerson

the ~~result~~ is with max_det=200, it not right because there maybe exists 800+ objects in a single object, so we re-evaluate(not finished) with max_det=1000 and got following result

detector | type | $AP_{50}^{tiny}$| script | COCO200 val $mmap$ | coco batch/lr
--- | --- | ---| ---| ---| ---
Faster-FPN | - |  ~~47.90~~<br/>49.81 | configs2/TinyPerson/base/Baseline_TinyPerson.sh:exp1.2 | - | -
Faster-FPN | SM | ~~50.06~~<br/>50.85 | ScaleMatch_TinyPerson.sh:exp4.0 | 18.9 | 8x2/0.01
Faster-FPN | SM | ~~49.53~~<br/>50.30 | ScaleMatch_TinyPerson.sh:exp4.1 | 18.5 | 4x2/0.01
Faster-FPN | MSM | ~~49.39~~<br/>50.18 | ScaleMatch_TinyPerson.sh:exp4.2 | 12.1 | 4x2/0.01
--| --| --
Adap RetainaNet-c | -   | ~~43.66~~<br/>45.22 | configs2/TinyPerson/base/Baseline_TinyPerson.sh:exp2.3 | - | -
Adap RetainaNet-c | SM  | ~~50.07~~<br/>51.78 | ScaleMatch_TinyPerson.sh:exp5.1 | 19.6 | 4x2/0.01
Adap RetainaNet-c | MSM | ~~48.39~~<br/>50.00 | ScaleMatch_TinyPerson.sh:exp5.2 | 12.9 | 4x2/0.01

for more experiment, to see [TinyPerson experiment](../../configs2/TinyPerson/TinyPerson.md)
for detail of scale match, to see [TinyPerson Scale Match](../../configs2/TinyPerson/scale_match/ScaleMatch.md)


## Citation

If you use the code and benchmark in your research, please cite:
```
@inproceedings{yu2020scale,
  title={Scale Match for Tiny Person Detection},
  author={Yu, Xuehui and Gong, Yuqi and Jiang, Nan and Ye, Qixiang and Han, Zhenjun},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1257--1265},
  year={2020}
}
```
And if the ECCVW challenge sumarry do some help for your research, please cite:
```
@article{yu20201st,
  title={The 1st Tiny Object Detection Challenge: Methods and Results},
  author={Yu, Xuehui and Han, Zhenjun and Gong, Yuqi and Jan, Nan and Zhao, Jian and Ye, Qixiang and Chen, Jie and Feng, Yuan and Zhang, Bin and Wang, Xiaodi and others},
  journal={arXiv preprint arXiv:2009.07506},
  year={2020}
}
```
