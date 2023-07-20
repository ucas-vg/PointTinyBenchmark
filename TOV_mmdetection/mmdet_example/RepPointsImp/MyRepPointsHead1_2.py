from MyRepPointsHead0 import MyRepPointsHead0
from mmcv.ops import DeformConv2d
from mmcv.cnn import ConvModule
import numpy as np
from torch import nn


class MyRepPointsHead1(MyRepPointsHead0):
    def __init__(self, num_classes, in_channels,
                 num_points=9,
                 point_feat_channels=256,
                 *args, **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels

        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = (self.dcn_kernel - 1) // 2
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'

        self.cls_out_channels = num_classes

        super().__init__(num_classes, in_channels, *args, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):  # stacked_convs=3
            chn = self.in_channels if i == 0 else self.feat_channels
            for module_list in [self.cls_convs, self.reg_convs]:
                module_list.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.conv_bias))
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1,
                                               self.dcn_pad)

        pts_out_dim = 2 * self.num_points
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *args, **kwargs):
        pass

    def loss(self, *args, **kwargs):
        pass

    def get_bboxes(self, *args, **kwargs):
        pass

    def get_targets(self, *args, **kwargs):
        pass


class MyRepPointsHead2(MyRepPointsHead1):
    def __init__(self, num_classes, in_channels,
                 num_points=9,
                 point_feat_channels=256,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',  # change here
                         std=0.01,
                         bias_prob=0.01)),
                 *args, **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels

        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = (self.dcn_kernel - 1) // 2
        super().__init__(num_classes, in_channels, init_cfg=init_cfg, *args, **kwargs)  # change here


if __name__ == '__main__':
    head = MyRepPointsHead2(
        num_classes=80,
        in_channels=256)
    print(head)
