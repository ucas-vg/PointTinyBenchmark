# import TOV_mmdetection.mmdet.models.dense_heads.condinst_head
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
# from mmdet.models.dense_heads.

@DETECTORS.register_module()
class BasicLocator(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BasicLocator,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_true_bboxes=None,
                      ):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, gt_true_bboxes)
        return losses
