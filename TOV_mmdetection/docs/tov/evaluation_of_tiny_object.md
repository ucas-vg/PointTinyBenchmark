# AP evaluation

There are two policy to crop image and merge results, taking TinyPerson as an example.

The results of the two ways may be little different because of different crop policy and max_per_img.

## way1: using origin annotation(run-time crop)

- crop image while loading image.
- merge results while inference.

config set as following

```python
test_pipeline = [
    ...
    dict(
        type='CroppedTilesFlipAug',
        tile_shape=(640, 512),  # sub image size by cropped
        tile_overlap=(100, 100),
        scale_factor=[1.0],
        flip=False,
    ...
    )
]

data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline)
)

model = dict(
    ...
    test_cfg=dict(
        nms_pre=2000,     # or try other setting
        max_per_img=1000  # or set as -1 or other
    ...
)
```

## way2: using corner annotation(offline crop)

- using corner annotation. (same as cutting image before evaluation)
- merge result after inference

to use such evaluation, need install mini_maskrcnn_benchmark
```
cd huicv
bash install.sh
```

config set as following

```python
test_pipeline = [
    ...
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,
    ...
    )
]

data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        merge_after_infer_kwargs=dict(
            merge_gt_file=data_root + 'mini_annotations/tiny_set_test_all.json',
            merge_nms_th=0.5
        ),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        merge_after_infer_kwargs=dict(
            merge_gt_file=data_root + 'mini_annotations/tiny_set_test_all.json',
            merge_nms_th=0.5
        ),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline)
)

model = dict(
    ...
    test_cfg=dict(
        nms_pre=1000,     # or try other setting
        max_per_img=500  # or set as -1 or other
    ...
)
```

- if you thought the evaluation is too much time-consuming, you can given a smaller *max_per_img*, 
such as 200 instead of 500. But it may get a bit lower evaluation result.

## evaluate on result file

For both two way, if you got an result file, and only need to evaluate on such file, you can run
```sh
# for result of corner images
python huicv/evaluation/evaluate_tiny.py --res exp/latest_result.json \
    --gt data/tiny_set/annotations/corner/task/tiny_set_test_sw640_sh512_all.json \
    --merge-gt data/tiny_set/mini_annotations/tiny_set_test_all.json --detail

# for result of origin images
python huicv/evaluation/evaluate_tiny.py --res exp/latest_result.json \
    --gt data/tiny_set/mini_annotations/tiny_set_test_all.json --detail

# reulst save in exp/scores.txt
```