The CPR code is in mmdet/models/dense_heads/cpr_head.py
and P2PNet code is in \mmdet\models\dense_heads\p2p_head.py

# Prerequisites
install environment following
```shell script
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

```shell script
conda install scikit-image  # or pip install scikit-image
```

# Test && Visualization

## Take COCO as example
1. move coco dataset or make a soft link to data/coco
2. download weight from [Baidu Yun(passwd:yuyx)](https://pan.baidu.com/s/1Fye7ZVINdkOR7xGvyTd1eg) or [Google Driver]() ,
move weights/CPR/coco/epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/epoch_12.pth
move weights/P2P/coco/CPR_epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/CPR_epoch_12.pth
move weights/P2P/coco/epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/epoch_12.pth

3. run such command for visualization of CPR
    ```shell script
    python tools/train.py configs2/COCO/coarsepointv2/coarse_point_refine_r50_fpn_1x_coco400_dbg.py \
        --work-dir ../TOV_mmdetection_cache/work_dir/tmp/ \
        --cfg-options \
            evaluation.do_first_eval=True \
            model.bbox_head.refine_pts_extractor.pos_generator.radius=8 \
            load_from="../TOV_mmdetection_cache/work_dir/coco/epoch_12.pth" \
            model.bbox_head.debug_info.COUNT=10 model.bbox_head.debug_info.epoch=-1 \
            model.bbox_head.debug_info.COUNT=10 model.bbox_head.debug_info.show=True
    ```
4. inference and visualization of P2PHead.
    ```shell script
    python demo/p2p_image_demo.py data/coco/images/000000005754.jpg \
     configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse.py \
     ../TOV_mmdetection_cache/work_dir/COCO/p2p_coarse/noise_rg-0-0-0.25-0.25_1/loss0gt_r8_8_lr0.01_1x_8b8g/_refine2_2_r8_8/p2p_r50_fpn_1x_fl_sl1_coco400_coarse/adam0.0001_1x_8b8g/n_epoch_12.pth
    ```
5. gif

6. different initial point to same refined point
 

## Take SeaPerson as example
3. run such command for visualization of CPR
    ```shell script
    python tools/train.py configs2/TinyPersonV2/coarsepointv2/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_dbg.py \
     --work-dir ../TOV_mmdetection_cache/work_dir/tmp/ \
     --cfg-options \
        evaluation.do_first_eval=True \
        model.bbox_head.refine_pts_extractor.pos_generator.radius=5 \
        load_from="../TOV_mmdetection_cache/work_dir/TinyPersonV2/p2p_coarse/noise_rg-0-0.25_1/loss0gt_r5_5_lr0.05_1x_2b4g/p2p_r50_fpns4_0.5x_fl_sl1_TinyPersonV2_640/adam0.0001_1x_2b4g/epoch_6.pth" \
        model.bbox_head.debug_info.COUNT=10 model.bbox_head.debug_info.epoch=-1 \
        model.bbox_head.debug_info.COUNT=10 model.bbox_head.debug_info.show=False
    ```

# Train Network

Learning rate set in config file is 4 times as that given in paper, 
while loss weight set in config file is 1/4 times of that given in paper.


## COCO
### Prepare dataset
1. download dataset to data/coco
2. generate point annotation or download point annotation(
[Baidu Yun passwd:1ej8](https://pan.baidu.com/s/11QDOZzognvZPaTgbJEYXAw) or 
[Google Driver](https://drive.google.com/drive/folders/1JHjvuSrgw7nCNDX3KSWpwJhaGGB3eUm-?usp=sharing)),
move annotations/coco/xxx to data/coco/xxx

### train 

1. P2PNet
    ```shell script
    GPU=8 && LR=0.0001 && B=8 && PORT=10000 tools/dist_train.sh configs2/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse.py ${GPU} \
            --work-dir ../TOV_mmdetection_cache/work_dir/COCO/p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse/adam${LR}_1x_${B}b${GPU}g${V}/ \
            --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B}
    ```

2. CPR + P2PNet
    ```shell script
    # [cmd 0] train CPRNet and inference on training set with CPRNet
    GPU=8 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh \
        configs2/COCO//coarsepointv2/coarse_point_refine_r50_fpn_1x_coco400.py 8 \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_r8_8_lr0.01_1x_8b8g/ \
        --cfg-options evaluation.save_result_file=../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_r8_8_lr0.01_1x_8b8g//latest_result_refine4_r8_8.json
    
    # [cmd 1] turn result file to coco annotation fmt
    python exp/tools/result2ann.py --ori_ann data/coco/coarse_gen_annotations/noise_rg-0-0-0.25-0.25_1/pseuw16h16/instances_train2017_coarse.json \
        --det_file ../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_r8_8_lr0.01_1x_8b8g//latest_result_refine4_r8_8.json \
        --save_ann ../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_r8_8_lr0.01_1x_8b8g//instances_train2017_refine4_r8_8.json
    
    # [cmd 2] train P2PNet
    export GPU=8 && export LR=0.0001 && export BATCH=8 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10001 tools/dist_train.sh \
        configs2/COCO//p2p/p2p_r50_fpn_1x_fl_sl1_coco400_coarse.py 8 \
        --work-dir ../TOV_mmdetection_cache/work_dir/COCO//p2p_coarse/noise_rg-0-0-0.25-0.25_1//loss0gt_r8_8_lr0.01_1x_8b8g/_refine4_r8_8/p2p_r50_fpn_1x_fl_sl1_coco400_coarse/adam0.0001_1x_8b8g/ \
        --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B} \
            data.train.ann_file=../TOV_mmdetection_cache/work_dir/COCO//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpn_1x_coco400/loss0gt_r8_8_lr0.01_1x_8b8g//instances_train2017_refine4_r8_8.json
    ```

## DOTA
### Prepare dataset
1. download dataset to data/dota
2. generate point annotation or download point annotation(
[Baidu Yun passwd:1ej8](https://pan.baidu.com/s/11QDOZzognvZPaTgbJEYXAw) or 
[Google Driver](https://drive.google.com/drive/folders/1JHjvuSrgw7nCNDX3KSWpwJhaGGB3eUm-?usp=sharing)),
move annotations/dota/xxx to data/dota/xxx

### train
1. P2PNet
    ```shell script
    
    ```

2. CPR + P2PNet
    ```shell script
    # [cmd 0]
    GPU=2 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh \
      configs2/DOTA//coarsepointv2/coarse_point_refine_r50_fpns4_1x_DOTA_1024.py 2 \
      --work-dir ../TOV_mmdetection_cache/work_dir/DOTA//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_DOTA_1024/loss0gt_r7_7_lr0.0001_1x_1b4g_s8/ \
      --cfg-options optimizer.lr=0.0001 data.samples_per_gpu=1 \
        model.bbox_head.strides=[8] model.neck.start_level=1 \
        model.bbox_head.refine_pts_extractor.pos_generator.radius=7 \
        model.bbox_head.refine_pts_extractor.neg_generator.radius=7  \
        model.bbox_head.train_pts_extractor.pos_generator.radius=7 \
        model.bbox_head.train_pts_extractor.neg_generator.radius=7 \
        evaluation.save_result_file=../TOV_mmdetection_cache/work_dir/DOTA//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_DOTA_1024/loss0gt_r7_7_lr0.0001_1x_1b4g_s8//latest_result_refine2_2_r7_7.json 
    
    # [cmd 1]
    python exp/tools/result2ann.py --ori_ann data/dota/DOTA-split/trainsplit/noise_rg-0-0-0.25-0.25_1/pseuw16h16/DOTA_train_1024_coarse.json \
      --det_file ../TOV_mmdetection_cache/work_dir/DOTA//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_DOTA_1024/loss0gt_r7_7_lr0.0001_1x_1b4g_s8//latest_result_refine2_2_r7_7.json \
      --save_ann ../TOV_mmdetection_cache/work_dir/DOTA//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_DOTA_1024/loss0gt_r7_7_lr0.0001_1x_1b4g_s8//DOTA_train_1024_refine2_2_r7_7.json
    
    # [cmd 2]
    export GPU=2 && export LR=0.0001 && export BATCH=4 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh \
      configs2/DOTA//p2p/p2p_r50_fpn_1x_fl_sl1_DOTA.py 2 \
      --work-dir ../TOV_mmdetection_cache/work_dir/DOTA//p2p_coarse/noise_rg-0-0-0.25-0.25_1//loss0gt_r7_7_lr0.0001_1x_1b4g_s8/p2p_r50_fpn_1x_fl_sl1_DOTA/adam0.0001_1x_4b2g_refine2_2_r7_7/  \
      --cfg-options optimizer.lr=0.0001 data.samples_per_gpu=4 \
        data.train.ann_file=../TOV_mmdetection_cache/work_dir/DOTA//coarsepointv2/noise_rg-0-0-0.25-0.25_1/coarse_point_refine_r50_fpns4_1x_DOTA_1024/loss0gt_r7_7_lr0.0001_1x_1b4g_s8//DOTA_train_1024_refine2_2_r7_7.json
    ```


## SeaPerson
### Prepare dataset
1. download dataset to data/seaperson # [TODO]
2. generate point annotation or download point annotation(
[Baidu Yun passwd:1ej8](https://pan.baidu.com/s/11QDOZzognvZPaTgbJEYXAw) or 
[Google Driver](https://drive.google.com/drive/folders/1JHjvuSrgw7nCNDX3KSWpwJhaGGB3eUm-?usp=sharing)),
move annotations/seaperson/xxx to data/seaperson/xxx

### train

1. P2PNet
    ```shell script
    export GPU=4 && LR=1e-4 && B=2 && WH=(640 640) && CONFIG="TinyPersonV2/p2p/p2p_r50_fpns4_1x_fl_sl1_TinyPersonV2_640" && \
    CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 tools/dist_train.sh configs2/${CONFIG}.py $GPU \
      --work-dir ../TOV_mmdetection_cache/work_dir/${CONFIG}/trainval${WH[0]}x${WH[1]}_adamlr${LR}_1x_b${B}${GPU}g_coarse/ \
      --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${B} \
        data.train.ann_file="data/tiny_set_v2/anns/release/corner/coarse/noise_rg-0-0.25_1/corner_w640_h640/pseuw16h16/rgb_train_w640h640ow100oh100_coarse.json" \
        data.val.ann_file="data/tiny_set_v2/anns/release/rgb_test.json"
    ```

2. CPR + P2PNet
    ```shell script
    # [cmd 0] train CPRNet and inference on training set with CPRNet
    GPU=4 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 tools/dist_train.sh \
        configs2/TinyPersonV2//coarsepointv2/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25.py 4 \
        --work-dir ../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r5_5_lr0.05_1x_2b4g/ 
        --cfg-options optimizer.lr=0.05 model.bbox_head.refine_pts_extractor.pos_generator.radius=5 model.bbox_head.refine_pts_extractor.neg_generator.radius=5  model.bbox_head.point_refiner.merge_th=0.1 model.bbox_head.point_refiner.refine_th=0.1 model.bbox_head.point_refiner.classify_filter=True model.bbox_head.train_pts_extractor.pos_generator.radius=5 model.bbox_head.train_pts_extractor.neg_generator.radius=5    model.bbox_head.loss_cfg.with_gt_loss=True      evaluation.save_result_file=../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r5_5_lr0.05_1x_2b4g//latest_result_refine2_2_r5_5.json
    
    # [cmd 1] turn result file to coco annotation fmt
    python exp/tools/result2ann.py --ori_ann data/tiny_set_v2/anns/release/corner/coarse/noise_rg-0-0.25_1/corner_w640_h640/pseuw16h16/rgb_train_w640h640ow100oh100_coarse.json --det_file ../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r5_5_lr0.05_1x_2b4g//latest_result_refine2_2_r5_5.json --save_ann ../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r5_5_lr0.05_1x_2b4g//rgb_train_w640h640ow100oh100_refine2_2_r5_5.json
    
    # [cmd 2] train P2PNet
    export GPU=4 && export LR=0.0001 && export BATCH=2 && export WH='(640 640)' && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 tools/dist_train.sh configs2/TinyPersonV2//p2p/p2p_r50_fpns4_0.5x_fl_sl1_TinyPersonV2_640.py 4 --work-dir ../TOV_mmdetection_cache/work_dir/TinyPersonV2//p2p_coarse/noise_rg-0-0.25_1//loss0gt_r5_5_lr0.05_1x_2b4g/p2p_r50_fpns4_0.5x_fl_sl1_TinyPersonV2_640/adam0.0001_1x_2b4g/ --cfg-options optimizer.lr=0.0001 data.samples_per_gpu=2 data.train.ann_file=../TOV_mmdetection_cache/work_dir/TinyPersonV2//coarsepointv2/noise_rg-0-0.25_1/coarse_point_refine_r50_fpns4_0.5x_TinyPersonV2_640_fixed_0_0.25/loss0gt_r5_5_lr0.05_1x_2b4g//rgb_train_w640h640ow100oh100_refine2_2_r5_5.json data.val.ann_file='data/tiny_set_v2/anns/release/rgb_test.json'
    ```


