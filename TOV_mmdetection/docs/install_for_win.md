
- msys2 (模仿shell)
- cl.exe (MVC编译器)
- conda: cuda/torch


1. [Install mmcv]()
[知乎教程](reference: https://zhuanlan.zhihu.com/p/308281195)

首先将更改为英语，控制面板->管理->更改区域设置->区域设置->英语(美国)

mklink /D data\shanghai_data E:\dataset\shanghai_data

```shell script
git clone https://github.com/open-mmlab/mmcv.git
git checkout v1.3.12 # based on target version

set MMCV_WITH_OPS=1
set MAX_JOBS=8
# change to your 
set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\"

# set TORCH_CUDA_ARCH_LIST=6.1 # 支持 GTX 1080
# 或者用所有支持的版本，但可能会变得很慢
# set TORCH_CUDA_ARCH_LIST=3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5
set TORCH_CUDA_ARCH_LIST=6.1 7.0 7.5

cd mmcv # 改变路径
python setup.py build_ext  # 如果成功, cl 将会自动弹出来编译 flow_warp
python setup.py develop
pip list
```

## problem
### 1. [CONSTEXPR_EXCEPT_WIN_CUDA](https://github.com/open-mmlab/mmcv/issues/575)
```
E:/software/Anaconda3/envs/torch160/lib/site-packages/torch/include\torch/csrc/jit/api/module.h(483): 
    error: a member with an in-class initializer must be const
```
将报错位置的 CONSTEXPR_EXCEPT_WIN_CUDA 修改为const.
If you look deeper at torch/include/c10/Macros.h, you will find CONSTEXPR_EXCEPT_WIN_CUDA is defined to empty, which causes the error.

