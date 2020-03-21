# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# add for freeanchor by hui
import torch


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=1. /9):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target, size_average=True):
        return smooth_l1_loss(input, target, self.beta, size_average)


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
