from maskrcnn_benchmark.layers import Scale
from torch import nn
import math
import torch


class LOCHead(nn.Module):
    def __init__(self, cfg, in_channels, no_centerness=False):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(LOCHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.LOC.NUM_CLASSES - 1
        self.no_centerness = no_centerness

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.LOC.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        if not no_centerness:
            self.centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )
            for l in self.centerness.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.LOC.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
            if not self.no_centerness:
                centerness.append(self.centerness(cls_tower))
        if self.no_centerness:
            centerness = None
        return logits, bbox_reg, centerness


def build_location_head(cfg, in_channel):
    return LOCHead(cfg, in_channel,
                   no_centerness=not (cfg.MODEL.LOC.TARGET_GENERATOR == 'fcos' and cfg.MODEL.LOC.FCOS_CENTERNESS))
