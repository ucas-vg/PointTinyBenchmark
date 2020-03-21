# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import logging
from torch.distributed import deprecated as dist

class AdjustSmoothL1Loss(nn.Module):

    def __init__(self, num_features, momentum=0.1, beta=1. /9):
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))
        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")

    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs -target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))


        beta = (self.running_mean - self.running_var)

        self.logger.info('AdjustSmoothL1(mean): {:.3}, {:.3}, {:.3}, {:.3}'.format(
            self.running_mean[0].item(),
            self.running_mean[1].item(),
            self.running_mean[2].item(),
            self.running_mean[3].item()
        ))
        self.logger.info('AdjustSmoothL1(var): {:.3}, {:.3}, {:.3}, {:.3}'.format(
            self.running_var[0].item(),
            self.running_var[1].item(),
            self.running_var[2].item(),
            self.running_var[3].item()
        ))
        beta = beta.clamp(max=self.beta, min=1e-3)

        #beta = (self.running_mean - self.running_var).clamp(
        #    max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

