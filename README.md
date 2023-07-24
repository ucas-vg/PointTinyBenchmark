[[paper]](figure/ICCV__SSD_DET.pdf) 

## Citation
If the work do some help for your research, please cite:

https://arxiv.org/abs/5016190

```
@inproceedings{SSDDET,
  author    = {Wu, Di and Chen, Pengfei and Yu, Xuehui and Li,
Guorong and Han, Zhenjun and Jiao, Jianbin},
  title     = {Spatial Self-Distillation for Object Detection with Inaccurate Bounding Boxes},
  booktitle = {ICCV},
  year      = {2023},
}
```

## Prerequisites

### 1. [install mmdetection](./docs/install.md>)
```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install -c pytorch pytorch=1.5.0 cudatoolkit=10.2 torchvision -y
# install the latest mmcv
pip install mmcv-full --user
# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

### 2. COCO dataset prepare

```bash
ln -s ${Path_Of_COCO} data/coco
```

## Train
#### Train on COCO

```bash
noise_level=0.4 \
&& cfg=configs/COCO/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_coco \
&& work_dir=../TOV_mmdetection_cache/COCO_n${noise_level}/${cfg##*/}/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10012 tools/dist_train.sh ${cfg}.py 8 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_objectness=True
```
#### Train on VOC
```bash
noise_level=0.4 \
&& cfg=configs/VOC/NoiseBox/ENoiseBoxPLUS_r50_fpn_1x_voc_${noise_level} \
&& work_dir=../TOV_mmdetection_cache/VOC_n${noise_level}/${cfg##*/}/ \
&& CUDA_VISIBLE_DEVICES=0,1 PORT=10012 tools/dist_train.sh ${cfg}.py 2 \
--work-dir=${work_dir} \
--cfg-options model.roi_head.with_objectness=True
```

