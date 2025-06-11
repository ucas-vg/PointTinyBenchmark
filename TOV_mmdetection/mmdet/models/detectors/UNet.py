# import TOV_mmdetection.mmdet.models.dense_heads.condinst_head
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
from mmdet.models.dense_heads.Unet_head import WeightedHausdorffDistance


@DETECTORS.register_module()
class Unet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Unet,
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
        losses = self.bbox_head.forward(x, gt_bboxes, gt_labels, img_metas)
        losses_final={}
        losses_final['loss_UNet']=losses
        # print(losses)
        return losses_final

    def simple_test(self, img, img_metas,  proposals=None, rescale=False):
        """Test without augmentation."""
        # base_proposal_cfg = self.train_cfg.get('base_proposal',
        #                                        self.test_cfg.rpn)
        # fine_proposal_cfg = self.train_cfg.get('fine_proposal',
        #                                        self.test_cfg.rpn)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        test_result = self.bbox_head.simple_test(x,
                                                                  img_metas,
                                                                  rescale=rescale)
        return test_result
