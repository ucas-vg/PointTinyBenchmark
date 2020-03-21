# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        # ############################### changed by hui #######################################################
        if key.startswith('roi_heads.box.predictor.') or key.startswith('rpn.head.'):
            lr *= cfg.SOLVER.HEAD_LR_FACTOR
        ########################################################################################################
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # ############################### changed by hui #######################################################
    if cfg.SOLVER.OPTIMIZER.lower() == 'adam':
        assert len(cfg.SOLVER.STEPS) == 0, 'solver.steps must set to empty when use adam as optimizer.'
        optimizer = torch.optim.Adam(params, lr, betas=cfg.SOLVER.ADAM_BETAS)
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    ########################################################################################################
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
