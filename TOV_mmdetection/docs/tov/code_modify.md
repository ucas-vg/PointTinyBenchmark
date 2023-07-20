### SUpport TinyPerson/visDronePerson Dataset

code modified as following:

function | file | necessary
---| --- | ---
coco format annotation + ignore dataset support | + mmdet\datasets\cocofmt.py<br/> > mmdet/datasets/__init__.py| Y
ScaleFactor=1.0 for ReSize; | > mmdet/datasets/pipelines/transforms.py:74,99,289| Y
corner dataset support| > mmdet/datasets/pipelines/loading.py:64 | X
auto cut image and merge result durring inference| + mmdet/datasets/pipelines/rtest_time_aug.py<br/> > mmdet/datasets/pipelines/__init__.py<br/> > mmdet/core/bbox/transforms.py:bbox_mapping,bbox_mapping_back<br/> > mmdet/models/dense_heads/dense_test_mixins.py:192<br/>mmdet/core/post_processing/merge_augs.py:69,102<br/> > mmdet/models/detectors/two_stage.py:aug_test+tile_aug_test</br> > mmdet/models/roi_heads/test_mixins.py:168,323| X
do final test| > do_final_eval：mmdet/core/evaluation/eval_hooks.py:10,40 | X
Scale Match| + mmdet/core/bbox/coder/bouding_box.py<br/>+ mmdet/datasets/pipelines/scale_match.py<br/> > mmdet/datasets/pipelines/__init__.py| X
stop while nan| > mmdet/apis/train.py:165,174;<br/> >${config}.py:add check=dict(stop_while_nan=True) | X
SparseRCNN  | > mmdet/models/roi_heads/sparse_roi_head.py:283 | X


**coco format annotation + ignore dataset support** include：
- load coco format json annotation;
- filte out 'ignore' bbox while load annotation;
- TinyPerson evaluation support;
- Location evaluation support;

```
evaluate function invoke graph

tools/train.py:train_detector 
-> mmdet/apis/train.py:eval_hook(,**cfg.get('evaluation')) 
-> mmdet/core/evaluation/eval_hooks.py:evaluate
-> mmdet/core/evaluation/eval_hooks.py:dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
-> mmdet/datasets/cocofmt.py:def evaluate(self,
    …,                
    cocofmt_kwargs={}):
    cocoEval = COCOExpandEval(cocoGt, cocoDt, iou_type, **cocofmt_kwargs)
```
