# add by hui ###################################
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import ConvTranspose2d
import torch
import numpy as np

LEAKY_RELU_RATE = 0.1


class CBRBlock(nn.Module):
    def __init__(self, in_channle, out_channle, kernel, stride, padding, no_relu=False, use_leaky_relu=False):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(in_channle, out_channle, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channle)
        self.no_relu = no_relu
        self.use_leaky_relu = use_leaky_relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        if not self.no_relu:
            if self.use_leaky_relu:
                x = F.leaky_relu(x, LEAKY_RELU_RATE)
            else:
                x = F.relu(x)
        return x


class EmptyBlock(object):
    def __call__(self, x):
        return x


class DeConvUpSampler(nn.Module):
    """
        up sample 4x
            deconv(2, 2, 0)
            transform():
                CBR: Conv BN Relu
                ...
                CBR
            deconv(2, 2, 0)
            transform()
    """
    def __init__(self, num_inputs=256, dim_reduced=256, num_conv=0, no_transform1=False, first_kernel=3,
                 no_relu=False, use_leaky_relu=False):
        super(DeConvUpSampler, self).__init__()
        self.first_kernel = first_kernel
        self.no_relu = no_relu
        self.use_leaky_relu = use_leaky_relu
        if no_transform1:
            self.transform1 = EmptyBlock()
        else:
            self.transform1 = self.build_transform(num_inputs, dim_reduced, dim_reduced, num_conv)
        self.deconv1 = ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
        self.transform2 = self.build_transform(dim_reduced, dim_reduced, dim_reduced, num_conv)
        self.deconv2 = ConvTranspose2d(dim_reduced, num_inputs, 2, 2, 0)

        for modules in [self.transform1.modules(), self.transform2.modules(), [self.deconv1, self.deconv2]]:
            for l in modules:
                if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(l.bias, 0)

    def build_transform(self, in_channel, reduce_channel, out_channel, num_conv):
        if num_conv <= 0:
            return EmptyBlock()
        if num_conv == 1:
            return nn.Sequential(CBRBlock(in_channel, out_channel, self.first_kernel, 1,
                                          (self.first_kernel-1)//2, self.no_relu, self.use_leaky_relu))
        transforms = [CBRBlock(in_channel, out_channel, self.first_kernel, 1,
                               (self.first_kernel-1)//2, self.no_relu, self.use_leaky_relu)]
        for i in range(num_conv-2):
            transforms.append(CBRBlock(reduce_channel, reduce_channel, 3, 1, 1, self.no_relu, self.use_leaky_relu))
        transforms.append(CBRBlock(reduce_channel, out_channel, 3, 1, 1, self.no_relu, self.use_leaky_relu))
        return nn.Sequential(*transforms)

    def forward(self, features):
        for i in range(len(features)):
            x = self.deconv1(self.transform1(features[i]))
            if not self.no_relu:
                if self.use_leaky_relu:
                    x = F.leaky_relu(x, LEAKY_RELU_RATE)
                else:
                    x = F.relu(x)
            features[i] = self.deconv2(self.transform2(x))
        return features


class InterpolateUpSampler(nn.Module):
    def __init__(self, upsample_rate, upsample_mode):
        super(InterpolateUpSampler, self).__init__()
        self.upsample_rate = upsample_rate
        self.upsample_mode = upsample_mode

    def forward(self, features):
        for i in range(len(features)):
            features[i] = F.interpolate(features[i], scale_factor=self.upsample_rate[i], mode=self.upsample_mode)
        return features


class PoolDownSampler(nn.Module):
    def __init__(self, downsample_rate):
        super(PoolDownSampler, self).__init__()
        self.down_sample_rate = downsample_rate

    def forward(self, features):
        for i in range(len(features)):
            s = self.down_sample_rate[i]
            features[i] = F.avg_pool2d(features[i], s, s, 0)
        return features


def build_sampler(cfg):
    upsample_rate = cfg.MODEL.UPSAMPLE_RATE
    upsample_mode = cfg.MODEL.UPSAMPLE_MODE  # add by hui
    upsample_transform_num_conv = cfg.MODEL.UPSAMPLE_TRANSFORM_NUM_CONV
    assert len(upsample_rate) == 0 or len(cfg.MODEL.FPN.UPSAMPLE_RATE) == 0, \
        'specified both cfg.MODEL.UPSAMPLE_RATE and cfg.MODEL.FPN.UPSAMPLE_RATE, it maybe a mistake.'
    if len(upsample_rate) == 0:
        return EmptyBlock()  # empty

    if upsample_mode == 'deconv':
        sampler = DeConvUpSampler(num_conv=upsample_transform_num_conv)
    elif upsample_mode == 'deconv2':
        sampler = DeConvUpSampler(num_conv=upsample_transform_num_conv, no_transform1=True)
    elif upsample_mode == 'deconv3':
        sampler = DeConvUpSampler(num_conv=upsample_transform_num_conv, no_relu=True)
    elif upsample_mode == 'deconv4':
        sampler = DeConvUpSampler(num_conv=upsample_transform_num_conv, first_kernel=1)
    elif upsample_mode == 'deconv5':
        sampler = DeConvUpSampler(num_conv=upsample_transform_num_conv, use_leaky_relu=True)
    elif upsample_mode == 'downsample':
        downsample_rate = torch.round(1 / torch.Tensor(upsample_rate)).numpy().astype(np.int).tolist()
        print("use downsample rate {}".format(downsample_rate))
        sampler = PoolDownSampler(downsample_rate)
    else:
        sampler = InterpolateUpSampler(upsample_rate, upsample_mode)

    return sampler

# ##############################################
