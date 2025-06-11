from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES

# sub-parts of the U-Net model

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(BaseModule):
    """"""
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True,
                 init_cfg=None):
        super(double_conv, self).__init__(init_cfg)

        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(BaseModule):
    def __init__(self, in_ch, out_ch, init_cfg=None):
        super(inconv, self).__init__(init_cfg)
        self.conv = double_conv(in_ch, out_ch, init_cfg=init_cfg)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(BaseModule):
    def __init__(self, in_ch, out_ch, normaliz=True, init_cfg=None):
        super(down, self).__init__(init_cfg)
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, normaliz=normaliz, init_cfg=init_cfg)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(BaseModule):
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True, init_cfg=None):
        super(up, self).__init__(init_cfg)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        # self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch,
                                normaliz=normaliz, activ=activ, init_cfg=init_cfg)

    def forward(self, x1, x2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Upsample is deprecated
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2)),
                        diffY // 2, int(math.ceil(diffY / 2))))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


@BACKBONES.register_module()
class LocateUNet(BaseModule):
    def __init__(self,
                 n_channels,
                 height, width,
                 ultrasmall=False,
                 device=torch.device('cuda'),
                 init_cfg=None):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param known_n_points: If you know the number of points,
                               (e.g, one pupil), then set it.
                               Otherwise it will be estimated by a lateral NN.
                               If provided, no lateral network will be build
                               and the resulting UNet will be a FCN.
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super().__init__(init_cfg)

        self.ultrasmall = ultrasmall
        self.device = device

        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        init_cfg = None
        self.inc = inconv(n_channels, 64, init_cfg=init_cfg)
        self.down1 = down(64, 128, init_cfg=init_cfg)
        self.down2 = down(128, 256, init_cfg=init_cfg)
        if self.ultrasmall:
            self.down3 = down(256, 512, normaliz=False, init_cfg=init_cfg)
            self.up1 = up(768, 128, init_cfg=init_cfg)
            self.up2 = up(256, 64, init_cfg=init_cfg)
            self.up3 = up(128, 64, activ=False, init_cfg=init_cfg)
        else:
            self.down3 = down(256, 512, init_cfg=init_cfg)
            self.down4 = down(512, 512, init_cfg=init_cfg)
            self.down5 = down(512, 512, init_cfg=init_cfg)
            self.down6 = down(512, 512, init_cfg=init_cfg)
            self.down7 = down(512, 512, init_cfg=init_cfg)
            self.down8 = down(512, 512, normaliz=False, init_cfg=init_cfg)
            self.up1 = up(1024, 512, init_cfg=init_cfg)
            self.up2 = up(1024, 512, init_cfg=init_cfg)
            self.up3 = up(1024, 512, init_cfg=init_cfg)
            self.up4 = up(1024, 512, init_cfg=init_cfg)
            self.up5 = up(1024, 256, init_cfg=init_cfg)
            self.up6 = up(512, 128, init_cfg=init_cfg)
            self.up7 = up(256, 64, init_cfg=init_cfg)
            self.up8 = up(128, 64, activ=False, init_cfg=init_cfg)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        else:
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x8 = self.down7(x7)
            x9 = self.down8(x8)
            x = self.up1(x9, x8)
            x = self.up2(x, x7)
            x = self.up3(x, x6)
            x = self.up4(x, x5)
            x = self.up5(x, x4)
            x = self.up6(x, x3)
            x = self.up7(x, x2)
            x = self.up8(x, x1)

        middle_layer = x4 if self.ultrasmall else x9
        return middle_layer, x


if __name__ == '__main__':
    model = BACKBONES.build(
        dict(
            type='LocateUNet',
            n_channels=3,
            n_classes=1,
            height=256,
            width=256,
            known_n_points=None,
            device=torch.device("cuda:0"),
            ultrasmall=False
             )
    )
    print(model)
    res = model(torch.randn(2, 3, 256, 256))
    (res[0].sum() + res[1].sum()).backward()
    print(res)
