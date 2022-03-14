

[comment]: <> (## Introduction)

[comment]: <> (TODO list:)

[comment]: <> (- [x] add TinyPerson dataset and evaluation)

[comment]: <> (- [x] add crop and merge for image during inference)

[comment]: <> (- [x] implement RetinaNet and Faster-FPN baseline on TinyPerson)

[comment]: <> (- [x] add SM/MSM experiment support)

[comment]: <> (<!-- - [ ] add visDronePerson dataset support and baseline performance)

[comment]: <> (- [ ] add point localization task for TinyPerson)

[comment]: <> (- [ ] add point localization task for visDronePerson)

[comment]: <> (- [ ] add point localization task for COCO -->)


## Install

### [install environment](./docs/install.md>)
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install -c pytorch pytorch=1.5.0 cudatoolkit=10.2 torchvision -y
# install the latest mmcv
pip install mmcv-full --user
```

```
conda install scikit-image
```

### download and project setting


- [note]: if your need to modified from origin mmdetection code, see [here](docs/tov/code_modify.md), otherwise do not need any other modified.
- [note]: for more about evaluation, see [evaluation_of_tiny_object.md](docs/tov/evaluation_of_tiny_object.md)

```shell script
git clone https://github.com/ucas-vg/PointTinyBenchmark # from github
# git clone https://gitee.com/ucas-vg/PointTinyBenchmark  # from gitee
cd PointTinyBenchmark/TOV_mmdetection
# download code for evaluation
git clone https://github.com/yinglang/huicv/  # from github
# git clone https://gitee.com/ucas-vg/huicv  # from gitee

# install mmdetection
pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
```

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
