import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class CascadeFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CascadeFCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        cascade_area_th = cfg.MODEL.FCOS.CASCADE_AREA_TH
        self.no_centerness = no_centerness = cfg.MODEL.FCOS.CASCADE_NO_CENTERNESS

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
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
        self.cls_logits_set = nn.ModuleDict()
        for area_th in cascade_area_th:
            self.cls_logits_set.add_module("cls_logits_{}%".format(int(area_th*100)), nn.Conv2d(
                in_channels, num_classes, kernel_size=3, stride=1, padding=1
            ))
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        if not no_centerness:
            self.centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.bbox_pred,  # self.centerness
                        ] + [m for m in self.cls_logits_set.values()]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for m in self.cls_logits_set.values():
            torch.nn.init.constant_(m.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits_set = {name: [] for name in self.cls_logits_set}
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            for name, cls_logits in self.cls_logits_set.items():
                logits_set[name].append(cls_logits(cls_tower))
            if not self.no_centerness:
                centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
        if len(centerness) == 0: centerness = None
        return logits_set, bbox_reg, centerness


class CascadeFCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(CascadeFCOSModule, self).__init__()

        head = CascadeFCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.vis_labels = cfg.MODEL.FCOS.DEBUG.VIS_LABELS

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls_set, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            res = self._forward_train(
                locations, box_cls_set,
                box_regression,
                centerness, targets
            )
        else:
            res = self._forward_test(
                locations, box_cls_set, box_regression,
                centerness, images.image_sizes, images=images, targets=targets
            )

        if self.vis_labels:
            show_image(images, targets, res[0])
        return res

    def _forward_train(self, locations, box_cls_set, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls_set, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
        }
        if isinstance(loss_centerness, torch.Tensor):
            losses["loss_centerness"] = loss_centerness
        return None, losses

    def _forward_test(self, locations, box_cls_set, box_regression, centerness, image_sizes, **kwargs):
        boxes = self.box_selector_test(
            locations, box_cls_set, box_regression,
            centerness, image_sizes, **kwargs
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_cascade_fcos(cfg, in_channels):
    return CascadeFCOSModule(cfg, in_channels)


class ResultShower(object):
    """
        1. plot image
        2. plot list of bboxes, bboxes can be ground-truth or detection results
        3. show score text for detection result
        4. show detection location as red point, score as point size
    """
    import numpy as np

    def __init__(self, image_mean=np.array([102.9801, 115.9465, 122.7717]), show_iter=1):
        self.score_th = None
        self.show_score_topk = 6
        self.image_mean = image_mean
        self.point_size = 100
        self.plot = self.plot2
        self.show_iter = show_iter
        self.counter = 0

    def __call__(self, images, *targets_list):
        import matplotlib.pyplot as plt
        import seaborn as sbn
        if (self.counter + 1) % self.show_iter != 0:
            self.counter += 1
            return
        self.counter += 1
        colors = sbn.color_palette(n_colors=len(targets_list))
        img = images.tensors[0].permute((1, 2, 0)).cpu().numpy() + self.image_mean
        img = img[:, :, [2, 1, 0]]
        plt.imshow(img/255)
        title = "boxes:"
        for ci, targets in enumerate(targets_list):
            if targets is not None:
                bboxes = targets[0].bbox.cpu().numpy().tolist()
                scores = targets[0].extra_fields['scores'].cpu() if 'scores' in targets[0].extra_fields else None
                locations = targets[0].extra_fields['det_locations'].cpu() if 'det_locations' in targets[0].extra_fields else None
                labels = targets[0].extra_fields['labels'].cpu()
                if scores is None:
                    self.plot1(bboxes, scores, locations, labels, None, (1, 0, 0))  # ground-truth
                else:
                    score_th = -torch.kthvalue(-scores, self.show_score_topk)[0]\
                        if self.score_th is None else self.score_th
                    self.plot(bboxes, scores, locations, labels, score_th, colors[ci])
                count = len(targets[0].bbox) if scores is None else (scores > score_th).sum()
                title += "{}({}) ".format(count, len(targets[0].bbox))
        plt.title(title)
        plt.show()
        input()

    def plot2(self, bboxes, scores, locations, labels, score_th, color=None):
        """
            no dash line link box and location, use color link
            different color for different box,
            same color for same box and location
        """
        import matplotlib.pyplot as plt
        import seaborn as sbn

        if True:# sorted
            scores, idx = (-scores).sort()
            scores = -scores
            labels = labels[idx]
            locations = locations[idx]

        colors = sbn.color_palette(n_colors=len(bboxes))
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            color = colors[i]
            if scores is not None:
                if scores[i] >= score_th:
                    plt.text(x1, y1, '{}:{:.2f}'.format(labels[i], scores[i]), color=(1, 0, 0))
                    rect = plt.Rectangle((x1, y1), w, h, fill=False, color=color, linewidth=1.5)
                    plt.axes().add_patch(rect)
                if locations is not None:
                    lx, ly = locations[i]
                    plt.scatter(lx, ly, color=color, s=self.point_size*scores[i])
            else:
                plt.text(x2, y2, '{}'.format(labels[i]), color=(1, 0, 0))
                rect = plt.Rectangle((x1, y1), w, h, fill=False, color=color, linewidth=1.5)
                plt.axes().add_patch(rect)

        print(scores)
        print(labels)
        print(locations)

    def plot1(self, bboxes, scores, locations, labels, score_th, color):
        """
        , use dash line link bbox and location
        """
        import matplotlib.pyplot as plt
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if scores is not None:
                if scores[i] >= score_th:
                    plt.text(x1, y1, '{:.2f}'.format(scores[i]), color=(1, 0, 0))
                    rect = plt.Rectangle((x1, y1), w, h, fill=False, color=color, linewidth=1.5)
                    plt.axes().add_patch(rect)
                    if locations is not None:
                        lx, ly = locations[i]
                        plt.plot([lx, lx, lx], [y2, ly, y1], '--', color=color)
                        plt.plot([x2, lx, x1], [ly, ly, ly], '--', color=color)
                if locations is not None:
                    lx, ly = locations[i]
                    plt.scatter(lx, ly, color='r', s=self.point_size * scores[i])
            else:
                rect = plt.Rectangle((x1, y1), w, h, fill=False, color=color, linewidth=1.5)
                plt.axes().add_patch(rect)


show_image = ResultShower()
