from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class MyRepPointsHead0(AnchorFreeHead):
    def __init__(self, num_classes, in_channels, *args, **kwargs):
        super(MyRepPointsHead0, self).__init__(num_classes, in_channels, *args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def loss(self, *args, **kwargs):
        pass

    def get_bboxes(self, *args, **kwargs):
        pass

    def get_targets(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    cfg = dict(
        type="MyRepPointsHead0",
        num_classes=80,
        in_channels=256
    )

    # 输出的是AnchorFreeHead搭建的网络
    head = HEADS.build(cfg, default_args=dict())
    print(head)
