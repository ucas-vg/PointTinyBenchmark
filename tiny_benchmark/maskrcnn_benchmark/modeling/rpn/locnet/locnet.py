import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_location_postprocessor
from .loss import make_location_loss_evaluator
from .head import build_location_head


class LocationGenerator(object):
    def __init__(self, fpn_strides):
        self.fpn_strides = fpn_strides

    def __call__(self, features):
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


class LOCModule(nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(LOCModule, self).__init__()
        self.head = build_location_head(cfg, in_channels)
        self.anchor_generator = LocationGenerator(cfg.MODEL.LOC.FPN_STRIDES)  # anchor or locations
        self.loss_evaluator = make_location_loss_evaluator(cfg)
        self.infer = make_location_postprocessor(cfg)

        self.debug_vis_labels = cfg.MODEL.LOC.DEBUG.VIS_LABELS

    def forward(self, images, features, targets=None):
        # preds: (box_cls, box_regression, centerness) for FCOS
        preds = self.head(features)
        # anchors for anchor-base, locations for anchor free
        anchors = self.anchor_generator(features)

        if self.debug_vis_labels:
            losses = self.loss_evaluator(anchors, *preds, targets)
            boxes = self.infer(anchors, *preds, images.image_sizes)
            show_image(images, targets, boxes)
            return boxes, {}
        else:
            if self.training:
                losses = self.loss_evaluator(anchors, *preds, targets)
                return None, losses
            else:
                boxes = self.infer(anchors, *preds, images.image_sizes)
                return boxes, {}


def build_location_net(cfg, in_channels):
    return LOCModule(cfg, in_channels)


import numpy as np
class ResultShower(object):
    """
        1. plot image
        2. plot list of bboxes, bboxes can be ground-truth or detection results
        3. show score text for detection result
        4. show detection location as red point, score as point size
    """

    def __init__(self, image_mean=np.array([102.9801, 115.9465, 122.7717]), show_iter=1):
        self.score_th = None
        self.show_score_topk = 4
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
                bboxes = targets[0].bbox.detach().cpu().numpy().tolist()
                scores = targets[0].extra_fields['scores'].detach().cpu() if 'scores' in targets[0].extra_fields else None
                locations = targets[0].extra_fields['det_locations'].detach().cpu() if 'det_locations' in targets[0].extra_fields else None
                labels = targets[0].extra_fields['labels'].cpu()
                if scores is None or len(scores) == 0:
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

        assert len(locations) == len(scores)

        if True:  # sorted
            scores, idx = (-scores).sort()
            scores = -scores
            labels = labels[idx]
            locations = locations[idx]
            bboxes = np.array(bboxes)[idx.numpy()]

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
        print(bboxes)

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
