The P2BNet code is in mmdet/models/detectors/P2BNet.py mmdet/models/roi_heads/P2B_head.py
our GUPs: 8 * RTX3090

# Prerequisites
install environment following
```shell script
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# install the latest mmcv

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    ```

# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

```shell script
conda install scikit-image  # or pip install scikit-image
```



#  Prepare dataset COCO
1. download dataset to data/coco
2. generate point annotation or download point annotation(
[Baidu Yun passwd:6752](https://pan.baidu.com/s/1XF9TneCxByqOJfAaqciP8A?pwd=6752 ) or 
[Google Driver]()],
move annotations/xxx to data/coco/annotations_qc_pt/xxx
3. or, you can generate annotation file by yourself:


#  QC Point generation for coco
## 1.generate QC point annotation
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
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json" \
    --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
    --size_range "${SR}"
```
## 2.Transfer QC point annotation to 'bbox' and transfer original bbox to 'true_bbox'
### the QC point annotation is transfered to 'bbox' with fixed w and h, which is easy for mmdetection reading and dataset pipeline
### the original bbox is transfered to 'true_bbox', which is the real box ground-turth
```sh
export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export RS=0.25
export CORNER=""
export WH=(64 64)
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json"  \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse_with_gt.json"  \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}
```

## 3.For other dataset, we can transform the annotation style to coco json style and use the same way.

# Train, Test and Visualization

## Take COCO as example
### Prepare trained model 
1. move coco dataset (2017 version) or make a soft link to data/coco
2. download weight from [Baidu Yun(passwd:3pfu)](https://pan.baidu.com/s/1G_S0zYJNMtBYF3fiH6XcKA?pwd=3pfu) or [Google Driver]() ,
move weights/P2BNet/epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/P2BNet/epoch_12.pth
move weights/faster/epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/Faster/epoch_12.pth



### Train 
```open to the work path: P2BNet/TOV_mmdetection```
1. P2BNet + FasterRCNN
    ```shell script
    # [cmd 0] train P2BNet and inference on training set with P2BNet
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
	--work-dir=${work_dir}  \
	--cfg-options evaluation.save_result_file=${work_dir}'_1200_latest_result.json'
	
    # [cmd 1] turn result file to coco annotation fmt
	python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json  ../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json
    
    # [cmd 2] train FasterRCNN
    	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 \
	--work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
    ```

2. inference with trained P2BNet to get pseudo box and train FasterRCNN with pseudo box
    ```shell script
    # [cmd 0] inference with trained P2BNet to get pseudo box
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
	--work-dir=${work_dir}  \
	--cfg-options  evaluation.save_result_file=${work_dir}'_1200_latest_result.json' load_from=${work_dir}'P2BNet/epoch_12.pth' evaluation.do_first_eval=True runner.max_epochs=0 
	
    # [cmd 1] turn result file to coco annotation fmt
	python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json ../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json  ../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json
    
    # [cmd 2] train FasterRCNN
    	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10003 ./tools/dist_train.sh configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py 8 \
	--work-dir=${work_dir}'detection/without_weight' --cfg-options data.train.ann_file=${work_dir}'coco_1200_latest_pseudo_ann_1.json'
    ```


### Test && Visualization
```Inference and Visualization of FasterRCNN```
1.  run such command for visualization of detection performance on P2BNet-FR
    ```shell script
    # move weights/faster/epoch_12.pth to ../TOV_mmdetection_cache/work_dir/coco/Faster/epoch_12.pth
    # [cmd 1] 'exp/visulization/image_demo_P2BNet_FR.py' is the entrance of the visualization project. 
	##Line 30 and 31 define the model's config file and the checkpoint file path.
	    ## config_file = 'configs2/COCO/detection/faster_rcnn_r50_fpn_1x_coco.py'
	    ## checkpoint_file = '../TOV_mmdetection_cache/work_dir/coco/Faster/epoch_12.pth'

	##Line 34, 35 and 36 define the images to be visualized. Line 37 defines the output folder.
	    ## ann_file = 'data/coco/annotations/instances_val2017.json'  # visualize: only val set
	    #### in_folder = '{image folder}'  # visualize: the image folder you need or 'data/coco/images/'
	    ## img_dir = 'data/coco/images/'
	    ## out_dir = '../TOV_mmdetection_cache/work_dirs/center_like/COCO/detection/'

	python exp/visulization/image_demo_P2BNet_FR.py
    ```


2. run such command for visualization of P2BNet
   ```shell script
    # [cmd 0] inference with trained P2BNet to get pseudo box
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/P2BNet/P2BNet_r50_fpn_1x_coco_ms.py 8 \
	--work-dir=${work_dir}  \
	--cfg-options model.roi_head.with_atten=False  model.roi_head.bbox_head.with_loss_pseudo=False evaluation.save_result_file=${work_dir}'_1200_latest_result.json' load_from=${work_dir}'P2BNet/epoch_12.pth' evaluation.do_first_eval=True runner.max_epochs=0 
	
    # [cmd 1] turn result file to coco annotation fmt
	python exp/tools/result2ann.py data/coco/annotations/instances_train2017.json '../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json'  '../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json'

    # [cmd 2] visualization of pseudo annotation as P2BNet result
	##set line 31 as pseudo anotation path: gt_path='../TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_1.json'	
	python exp/visulization/vis_p2bnet.py 
    ```







