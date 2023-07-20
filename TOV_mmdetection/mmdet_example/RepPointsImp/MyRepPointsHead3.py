from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
import torch
import numpy as np
from MyRepPointsHead1_2 import MyRepPointsHead2


class MyRepPointsHead3(MyRepPointsHead2):
    def __init__(self, num_classes, in_channels,
                 gradient_mul=0.1,
                 *args, **kwargs):
        super().__init__(num_classes, in_channels, *args, **kwargs)

        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_x, dcn_base_y = np.meshgrid(dcn_base, dcn_base)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=-1).reshape((-1, 2))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.gradient_mul = gradient_mul

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        cls_feat, pts_feat = x, x
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat = cls_conv(cls_feat)
            pts_feat = reg_conv(pts_feat)

        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))

        pts_out_init_regrad = self.gradient_mul * pts_out_init + (1 - self.gradient_mul) * pts_out_init.detach()
        dcn_offset = pts_out_init_regrad - dcn_base_offset

        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()  # ******************** star 1
        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        return cls_out, pts_out_init, pts_out_refine


def assert_same(l1, l2):
    if isinstance(l1, (tuple, list)) or isinstance(l2, (tuple, list)):
        for x1, x2 in zip(l1, l2):
            return assert_same(x1, x2)
    else:
        assert (l1 - l2).sum() == 0, f"{l1, l2}"
    return True


if __name__ == '__main__':
    my_model = MyRepPointsHead3(
        num_classes=80,
        in_channels=256).to(0)

    lib_model = HEADS.build(
        cfg=dict(
            type='RepPointsHead',
            num_classes=80,
            in_channels=256,
            transform_method='minmax'
        )
    ).to(0)

    my_model.init_weights()
    from mmcv.runner.checkpoint import get_state_dict

    state_dict = get_state_dict(my_model)
    lib_model.load_state_dict(state_dict)

    # block
    # check our code with RepPointsHead implement in official library
    feats = [torch.randn(2, 256, 8, 8).to(0)]
    res1 = my_model(feats)
    res2 = lib_model(feats)
    print(assert_same(res1, res2))
