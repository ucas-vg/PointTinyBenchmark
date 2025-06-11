from .anchor_free_head import AnchorFreeHead
from ..builder import HEADS
from mmdet.core import multi_apply


@HEADS.register_module()
class RepPointLocHead(AnchorFreeHead):
    def __init__(self):
        pass

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        pass
