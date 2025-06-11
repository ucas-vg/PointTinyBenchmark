
1. add gt_xxx for get_bboxes for P2P

1.1 dataset

config file:
- add LoadAnnotations
- replace ImageToTensor with DefaultFormatBundle
- add 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes' to keys of Collect
- data.test_mode=False
```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # add
    dict(
        type='MultiScaleFlipAug',
        img_scale=(167, 100),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),  # add
            # dict(type='ImageToTensor', keys=['img']), # remove
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),  # add
        ])
]
data = dict(
    val=dict(
        pipeline=test_pipeline,
        test_mode=False  # add
    ),
)
```
mmdet/apis/train.py
```python
val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
=>
val_dataset = build_dataset(cfg.data.val, dict(test_mode=cfg.data.val.pop('test_mode', True)))
```

1.2 Detector: 添加gt_xxx的参数

- detector.forward_test
- detector.simple_test/async_simple_test
- detector.aug_test

BaseDetector mmdet/models/detectors/base.py
```python
    return await self.async_simple_test(img[0], img_metas[0], **kwargs)
=>
    for key in kwargs:  # modified by hui
        if key in ['proposals'] or key.startswith('gt_'):
            kwargs[key] = kwargs[key][0]
    return await self.async_simple_test(img[0], img_metas[0], **kwargs)
```
```python
    if 'proposals' in kwargs:
        kwargs['proposals'] = kwargs['proposals'][0]
=>
    for key in kwargs:  # modified by hui
        if key in ['proposals'] or key.startswith('gt_'):
            kwargs[key] = kwargs[key][0]
    return self.simple_test(imgs[0], img_metas[0], **kwargs)
```

SingleStageDetector mmdet/models/detectors/single_stage.py
```python
    results_list = self.bbox_head.simple_test(
        feat, img_metas, rescale=rescale)
=>
    gt_kwargs = {k: v for k, v in kwargs.items() if k.startswith('gt_')}
    results_list = self.bbox_head.simple_test(
        feat, img_metas, rescale=rescale, **gt_kwargs)
```

```python
    results_list = self.bbox_head.aug_test(
        feat, img_metas, rescale=rescale)
=>
    gt_kwargs = {k: v for k, v in kwargs.items() if k.startswith('gt_')}
    results_list = self.bbox_head.aug_test(
        feat, img_metas, rescale=rescale, **gt_kwargs)
```

1.3 Head: 添加**kwargs

- head.simple_test
- head.aug_test

AnchorFreeHead mmdet/models/dense_heads/anchor_free_head.py
```python
    def aug_test(self, feats, img_metas, rescale=False):
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
=>
    def aug_test(self, feats, img_metas, rescale=False, **kwargs):
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale, **kwargs)
```

BaseDenseHead mmdet/models/dense_heads/base_dense_head.py
```python
    def simple_test(self, feats, img_metas, rescale=False):
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)
=>
    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale, **kwargs)
```

BBoxTestMixin mmdet/models/dense_heads/dense_test_mixins.py
- simple_test_bboxes
- aug_test_bboxes
- simple_test_rpn
- async_simple_test_rpn

```python
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        ...
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
=>
    def simple_test_bboxes(self, feats, img_metas, rescale=False, **kwargs):
        ...
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, **kwargs)
```
