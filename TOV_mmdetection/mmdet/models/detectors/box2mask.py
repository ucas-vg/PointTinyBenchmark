from ..builder import DETECTORS
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Box2Mask(MaskFormer):
    r"""Implementation of `Box2Mask: Box-supervised Instance
    Segmentation via Level-set Evolution
    <https://arxiv.org/pdf/2212.01579.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)