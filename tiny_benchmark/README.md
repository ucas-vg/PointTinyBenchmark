### [Installation](#1.)
### [Getting started](#2.)
### [Evaluation](#3.)
### [Experiment](#4.)

---

# Installation <a name='1.'/>

### Requirements:
- PyTorch 1.0.1
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setuped properly with the right environment
# for that, check `which conda`, `which pip` and `which python`, whether they point to the
# right path. From a new conda env, the following is what you need to do:

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# install the right pip and dependencies
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install -r requirements.txt

# PyTorch installation

# PyTorch installation
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch

conda install opencv
conda install scipy

# install pycocotools
cd ${TinyBenchmark}/tiny_benchmark
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../../

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
# rm build # if needed
python setup.py build develop

# or if you use MacOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```


### normal issuse

> A runtime bug: undefined symbol: _ZN3c105ErrorC1ENS_14SourceLocationERKSs

```bash
    from maskrcnn_benchmark.layers import nms as _box_nms  File "/home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/layers/__init__.py", line 9, in <module>

  File "/home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/layers/__init__.py", line 9, in <module>
    from .nms import nms
  File "/home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/layers/nms.py", line 3, in <module>
    from .nms import nms
  File "/home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/layers/nms.py", line 3, in <module>
    from maskrcnn_benchmark import _C
ImportError:     /home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/_C.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC1ENS_14SourceLocationERKSsfrom maskrcnn_benchmark import _C

ImportError: /home/data/github/tiny_benchmark/tiny_benchmark/maskrcnn_benchmark/_C.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC1ENS_14SourceLocationERKSs
```


- solution:

change the code in $YOU_CONDA_ENV_DIR/lib/python3.6/site-packages/torch/utils/cpp_extension.py as follows:

```py
-D_GLIBCXX_USE_CXX11_ABI=0
=>
-D_GLIBCXX_USE_CXX11_ABI=1
```


which are in line 398 and line 1013

```py
self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=0')
...
common_cflags += ['-D_GLIBCXX_USE_CXX11_ABI=0']
=>
self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=1')
...
common_cflags += ['-D_GLIBCXX_USE_CXX11_ABI=1']
```


and then delete ‘build’ and ‘re-build’

```sh
rm build -rf
python setup.py build develop
```


- possible reason:

The gcc flag should keep the same, while build pytorch and it's extension.
The possible cause is previous pytorch version use -D_GLIBCXX_USE_CXX11_ABI=0 to build in conda,
therefore it's ok to build nms extension using -D_GLIBCXX_USE_CXX11_ABI=0, however the new version 
build with -D_GLIBCXX_USE_CXX11_ABI=1, resulting in this bug.

# Getting started <a name='2.'/>
1. install TinyBenchamrk [Install]()
2. download dataset [dataset](../dataset) and move to \\\${TinyBenchmark}/dataset,
modify path in \\\${TinyBenchmark}/tiny_benchmark/maskrcnn_benchmark/config/paths_catalog.py to your dataset path

```py
class DatasetCatalog(object):
    DATA_DIR = "/home/$user/TinyBenchmark/dataset"
    DATASETS = {
    ....
```
3. download our Scale Match COCO pretrain weight, see [here](../params/Readme.md);<br/>
Or if you want to train your own Scale Match COCO pretrain weight, see [here](configs/TinyCOCO/Readme.md)

4. choose a config file and run as [maskrcnn_benchmark training](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training)，<font color='ff0000'/>**Attention!!! use "tools/train_test_net.py" instead of "tools/train_net.py"**</font>

perhaps you need to change change MERGE_GT_FILE in config file to gt file path your downlaoded.
```
TEST:
  IMS_PER_BATCH: 2
  COCO_EVALUATE_STANDARD: 'tiny'  # tiny need change
  MERGE_RESULTS: true
  MERGE_GT_FILE: '/home/hui/dataset/tiny_set/annotations/task/tiny_set_test_all.json'  # change to your gt file path
  IGNORE_UNCERTAIN: true
  USE_IOD_FOR_IGNORE: true
```

train and test
```sh
cd ${TinyBenchmark}/tiny_benchmark
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=9001 tools/train_test_net.py --config ${config_path}
```

~~Notice: the test annotation will not be released until RLQ-TOD@ECCV'20 challenge finished, you may need to change DATASETS.TEST in config file for training, such as:~~
~~```yaml~~
~~DATASETS:~~
~~  TRAIN: ("tiny_set_corner_sw640_sh512_erase_with_uncertain_train_all_coco",)~~
~~  TEST: ("tiny_set_corner_sw640_sh512_erase_with_uncertain_train_all_coco",)~~
~~```~~

# Evaluation <a name='3.'/>

if you use tiny_benchmark to train, the evaluate will run auto. But if you use other code to generate result, you can use script **tiny_benchamrk/MyPackage/tools/evaluate_tiny.py** to evalute, for example
```sh
python evaluate_tiny.py --res ~/results/your_result.json --gt ~/dataset/tiny_set_test_all.json --detail
```
and the mr evaluation is time-cost, you can specified --metric to evaluate AP only
```sh
python evaluate_tiny.py --res ~/results/your_result.json --gt ~/dataset/tiny_set_test_all.json --detail --metric 'ap'
```

Moreover your_result.json should satisfy such format
```
[{'image_id': 793, 'category_id': 1, 'bbox': [0.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582}, 
{'image_id': 793, 'category_id': 1, 'bbox': [590.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582},
......
{'image_id': 795, 'category_id': 1, 'bbox': [1180.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582}]
```

~~You can split a sub-set from training setting to evalute you model. For evalution on the test set,~~ 
And you can upload your result to [ECCVW challenge (RLQ-TOP@ECCV'20)](https://competitions.codalab.org/competitions/24551#results).

# Experiment <a name='4.'/>

<a color='#00ff00'> Notice: in following tables, **updated evaluation code (compared with the WACV paper) was adopted**. Since orginal code for wacv paper handles the ignore region not well, we have updated the evaluation code and obtained some new experimental results. Although the modification for evaluation, the relevant conclusions are consistent. Each group of experiments was run at least 3 times, and the final experimental result was the average of multiple results.</a>

For details of experiment setting, please see [paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.pdf) Section 5.1. Experiments Setting

training setting| value
---|---
training imageset| dataset/tiny_set/erase_with_uncertain_dataset/train
training annotation| dataset/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json
test annotation| dataset/tiny_set/annotations/task/tiny_set_test_all.json
deal to ignore region while training| erase with mean color
size of cut image piece| (640, 512)

## 1. detectors

- <font color='red'>note: for FCOS, if remove GN in head, it can achieve comparable performance with other framework. see [our mmdetection version](https://github.com/ucas-vg/TOV_mmdetection/blob/main/configs2/TinyPerson/TinyPerson.md#2-performance)</font>

detector | $AP^{tiny1}_{50}$ | $AP^{tiny2}_{50}$ |  $AP^{tiny3}_{50}$ | $AP^{tiny}_{50}$ | $AP^{small}_{50}$| $AP^{tiny}_{25}$| $AP^{tiny}_{75}$
---|---|---|---|---|---|---|---
[FCOS](configs/TinyPerson/fcos/baseline1/fcos_R_50_FPN_1x_baseline1.yaml) 						| 0.99 | 2.82 | 6.2 | 3.26 | 20.19 | 13.28 | 0.14
[RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lr.yaml)			| 12.24 | 38.79 | 47.38 | 33.53 | 48.26 | 61.51 | 2.28
DSFD                   | 13.85| 37.24| 49.31| 33.65 | 56.64 | 63.18| 1.94
[Adaptive RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml)                              | 27.08 | 52.63 | 57.88 | 46.56 | 59.97 | 69.6 | 4.49
[Adaptive FreeAnchor](configs/TinyPerson/freeanchor/baseline1/freeanchor_R_50_FPN_1x_baseline1_lrfpn.yaml) | 25.13 | 47.41 | 52.77 | 41.41 | 59.61 | 63.38 | 4.58
[Faster RCNN-FPN](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)       | 30.25|51.58|58.95|**47.35**|63.18|68.43|5.83

detector | $MR^{tiny1}_{50}$ | $MR^{tiny2}_{50}$ |  $MR^{tiny3}_{50}$ | $MR^{tiny}_{50}$ | $MR^{small}_{50}$ | $MR^{tiny}_{25}$ | $MR^{tiny}_{75}$
---|---|---|---|---|---|---|---
[FCOS](configs/TinyPerson/fcos/baseline1/fcos_R_50_FPN_1x_baseline1.yaml) 					  | 99.96 | 99.77 | 97.68 | 99.0 | 95.49 | 97.24 | 99.89
[RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lr.yaml)			     | 94.52 | 88.24 | 86.52 | 92.66 | 82.84 | 81.95 | 99.13
DSFD                      | 96.41| 88.02| 86.84| 93.47| 78.72| 78.02| 99.48
[Adaptive RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml)    |  89.65 | 81.03 | 81.08 | 88.31 | 74.05 | 76.33 | 98.76
[Adaptive FreeAnchor](configs/TinyPerson/freeanchor/baseline1/freeanchor_R_50_FPN_1x_baseline1_lrfpn.yaml) | 88.93 | **80.75** | 83.63 | 89.63 | 74.38 | 78.21 | 98.77
[Faster RCNN-FPN](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)    | 87.86|82.02|78.78|**87.57**|72.56|76.59|98.39 

### The experimental results based on the orginal evaluation code in WACV paper (Not recommended)

detector | $AP^{tiny1}_{50}$ | $AP^{tiny2}_{50}$ |  $AP^{tiny3}_{50}$ | $AP^{tiny}_{50}$ | $AP^{small}_{50}$| $AP^{tiny}_{25}$| $AP^{tiny}_{75}$
---|---|---|---|---|---|---|---
FCOS | 3.39|12.39|29.25|16.9|35.75|40.49|1.45
RetinaNet | 11.47|36.36|43.32|30.82|43.38|57.33|2.64
DSFD | 14.00|35.12|46.30|31.15|51.64|59.58|1.99
Adaptive RetinaNet | 25.49|47.89|51.28|41.25|53.64|63.94|4.22
Adaptive FreeAnchor | 24.92|48.01|51.23|41.36|53.36|63.73|4.0
Faster RCNN-FPN | 29.21|48.26|53.48|43.55|56.69|64.07|5.35

detector | $MR^{tiny1}_{50}$ | $MR^{tiny2}_{50}$ |  $MR^{tiny3}_{50}$ | $MR^{tiny}_{50}$ | $MR^{small}_{50}$ | $MR^{tiny}_{25}$ | $MR^{tiny}_{75}$
---|---|---|---|---|---|---|---
FCOS | 99.10|96.39|91.31|96.12|84.14|89.56|99.56
RetinaNet | 95.05|88.34|86.04|92.40|81.75|81.56|99.11
DSFD | 96.41|88.02|86.84|93.47|78.72|78.02|99.48
Adaptive RetinaNet | 89.48|82.29|82.40|89.19|74.29|77.83|98.63
Adaptive FreeAnchor | 90.26|82.01|81.74|88.97|73.67|77.62|98.7
Faster RCNN-FPN | 88.40|81.99|80.17|87.78|71.31|77.35|98.40
        
## 2. scale match

### 2.1 Scale Match on FPN, TinyPerson

pretrain dataset| ap50_tiny1|ap50_tiny2|ap50_tiny3|ap50_tiny|ap50_small|ap25_tiny|ap75_tiny
---| ---|---|---|---|---|---|---
[ImageNet](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)            |30.25|51.58|58.95|47.35|63.18|68.43|5.83
[COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco.yaml) |28.04|50.67|59.83|47.11|66.44|67.81|6.25
[COCO$^{A}$](configs/TinyPerson/FPN/baseline2/v3/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco_v3.yaml)              | 31.69 | 54.08 | 61.24 | 49.76 | 66.46 | 70.91 | 6.58
[COCO100](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco100.yaml) | 32.1 | 52.6 | 58.57 | 47.73 | 63.7 | 69.26 | 5.91
COCO100$^{A}$   | 32.21 | 53.71 | 61.56 | 49.57 | 66.44 | 70.24 | 6.1
[SM COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_smb4coco.yaml)          |**33.91**|**55.16**|**62.58**|**51.33**|**66.96**|**71.55**|6.46
[MSM COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_msmb4coco.yaml)                 |33.79|55.55|61.29|50.89|65.76|71.28|**6.66**

table| mr50_tiny1|mr50_tiny2|mr50_tiny3|mr50_tiny|mr50_small|mr25_tiny|mr75_tiny
---| ---|---|---|---|---|---|---
[ImageNet](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)               |87.86|82.02|78.78|87.57|72.56|76.59|98.39
[COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco.yaml) |88.19|81.08|76.84|86.24|67.55|76.08|98.14
[COCO$^{A}$](configs/TinyPerson/FPN/baseline2/v3/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco_v3.yaml)    | 87.82 | 80.16 | 77.22 | 86.85 | 69.43 | 75.71 | 98.32
[COCO100](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_coco100.yaml) |88.14|81.94|78.97|87.64|72.55|76.62|98.37
COCO100$^{A}$ |86.94|80.53|77.06|86.4|69.88|75.44|98.2
[SM COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_smb4coco.yaml)     | 87.14 | 79.60 |**76.14**|86.22 |**68.59**|**74.16**|98.28
[MSM COCO](configs/TinyPerson/FPN/baseline2/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_msmb4coco.yaml)       |**86.54**|**79.2**|76.86|**85.86**|68.76|74.33|**98.23**

X$^{A}$(X is COCO or COCO100) means we use the different anchor setting as pre-train while finetine, but keep same with SM COCO. In WACV paper, there are no such experiments.

### 2.2 Scale Match on Adaptive RetinaNet, TinyPerson

table| ap50_tiny1|ap50_tiny2|ap50_tiny3|ap50_tiny|ap50_small|ap50_all|ap25_tiny|ap75_tiny
---| ---|---|---|---|---|---|---|---
[ImageNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml) | 27.08 | 52.63 | 57.88 | 46.56 | 59.97 | 48.4 | 69.6 | 4.49
COCO$^{A}$ | 25.36 | 52.41 | 55.6 | 45.03 | 59.49 | 47.51 | 68.07 | 4.75
[SM COCO](configs/TinyPerson/retina/baseline2/retina_R_50_FPN_1x_baseline2_lrfpn_smb4coco.yaml) | 29.01 | 54.28 | 59.95 | 48.48 | 63.01 | 51.06 | 69.41 | 5.83
[MSM COCO](configs/TinyPerson/retina/baseline2/retina_R_50_FPN_1x_baseline2_lrfpn_msmb4coco.yaml)  | **31.63** | **56.01** | **60.78** | **49.59** | **63.38** | **51.75** | **71.24** | **6.16**

table| mr50_tiny1|mr50_tiny2|mr50_tiny3|mr50_tiny|mr50_small|mr50_all|mr25_tiny|mr75_tiny
---| ---|---|---|---|---|---|---|---
[ImageNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml) | 89.65 | 81.03 | 81.08 | **88.31** | 74.05 | 90.53 | 76.33 | 98.76
COCO$^{A}$ | 90.38 | 82.9 | 82.61 | 89.42 | 74.57 | 91.0 | 77.2 | 98.77
[SM COCO](configs/TinyPerson/retina/baseline2/retina_R_50_FPN_1x_baseline2_lrfpn_smb4coco.yaml) | 89.83 | 81.19 | 80.89 | 88.87 | **71.82** | 89.63 | 77.88 | **98.57**
[MSM COCO](configs/TinyPerson/retina/baseline2/retina_R_50_FPN_1x_baseline2_lrfpn_msmb4coco.yaml) | **87.8** | **79.23** | **79.77** | 88.39 | 72.18 | **89.4** | **76.25** | 98.57

```{.python .input}

```
