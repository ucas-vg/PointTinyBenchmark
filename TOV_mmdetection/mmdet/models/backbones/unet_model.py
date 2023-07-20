__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .unet_parts import *
from ..builder import BACKBONES

@BACKBONES.register_module()
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=80,
                 height=400, width=667,
                 known_n_points=None,
                 ultrasmall=False,
                 pretrained=None,
                 device=torch.device('cuda')):
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
        super(UNet, self).__init__()

        self.ultrasmall = ultrasmall
        self.device = device

        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        if self.ultrasmall:
            self.down3 = down(256, 512, normaliz=False)
            self.up1 = up(768, 128)
            self.up2 = up(256, 64)
            self.up3 = up(128, 64, activ=False)
        else:
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.down5 = down(512, 512)
            self.down6 = down(512, 512)
            self.down7 = down(512, 512)
            self.down8 = down(512, 512, normaliz=False)
            self.up1 = up(1024, 512)
            self.up2 = up(1024, 512)
            self.up3 = up(1024, 512)
            self.up4 = up(1024, 512)
            self.up5 = up(1024, 256)
            self.up6 = up(512, 128)
            self.up7 = up(256, 64)
            self.up8 = up(128, 64, activ=False)
        self.outc = outconv(64, n_classes)
        self.out_nonlin = nn.Sigmoid()

        self.known_n_points = known_n_points
        if known_n_points is None:
            steps = 3 if self.ultrasmall else 8
            height_mid_features = height//(2**steps)
            width_mid_features = width//(2**steps)
            self.branch_1 = nn.Sequential(nn.Linear(height_mid_features*\
                                                    width_mid_features*\
                                                    512,
                                                    64),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.5))
            self.branch_2 = nn.Sequential(nn.Linear(height*width, 64),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.5))
            self.regressor = nn.Sequential(nn.Linear(64 + 64, 1),
                                           nn.ReLU())

        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):

        batch_size = x.shape[0]

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
        x = self.outc(x)
        x = self.out_nonlin(x)

        # Reshape Bx1xHxW -> BxHxW
        # because probability map is real-valued by definition
        x = x.squeeze(1)
        return x
        # if self.known_n_points is None:
        #     middle_layer = x4 if self.ultrasmall else x9
        #     middle_layer_flat = middle_layer.view(batch_size, -1)
        #     x_flat = x.view(batch_size, -1)
        #
        #     lateral_flat = self.branch_1(middle_layer_flat)
        #     x_flat = self.branch_2(x_flat)
        #
        #     regression_features = torch.cat((x_flat, lateral_flat), dim=1)
        #     regression = self.regressor(regression_features)
        #
        #     return x, regression
        # else:
        #     n_pts = torch.tensor([self.known_n_points]*batch_size,
        #                          dtype=torch.get_default_dtype())
        #     n_pts = n_pts.to(self.device)
        #     return x, n_pts
        # summ = torch.sum(x)
        # count = self.lin(summ)

        # count = torch.abs(count)

        # if self.known_n_points is not None:
            # count = Variable(torch.cuda.FloatTensor([self.known_n_points]))

        # return x, count


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019 
"""
