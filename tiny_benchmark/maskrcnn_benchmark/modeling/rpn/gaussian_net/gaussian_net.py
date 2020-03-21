import torch
from torch import nn
import math
from .loss import make_gau_loss_evaluator
from .inference import make_gau_postprocessor


class GauHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(GauHead, self).__init__()
        num_classes = cfg.MODEL.GAU.NUM_CLASSES - 1

        # cls_tower
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.GAU.NUM_CONVS):
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

        # cls_logits
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.use_more_logits = True
        if self.use_more_logits:
            self.gau_logits = nn.Conv2d(
                in_channels, num_classes, kernel_size=3, stride=1, padding=1
            )
            # self.dif_logits = nn.Conv2d(
            #     in_channels, num_classes, kernel_size=3, stride=1, padding=1
            # )

        # upsample towers
        # up_towers = []
        # for i, stride in enumerate(cfg.MODEL.GAU.FPN_STRIDES):
        #     # up_tower = []
        #     # for _ in range(int(math.log2(stride)) - 2):
        #     #     up_tower.append(nn.UpsamplingBilinear2d(scale_factor=2))
        #     #     up_tower.append(
        #     #         nn.Conv2d(
        #     #             num_classes,
        #     #             num_classes,
        #     #             3, 1, 1, groups=num_classes
        #     #         )
        #     #     )
        #     #     up_tower.append(nn.ReLU())
        #     #     # up_tower.append(
        #     #     #     nn.ConvTranspose2d(
        #     #     #         num_classes,
        #     #     #         num_classes,
        #     #     #         kernel_size=3,
        #     #     #         stride=2,
        #     #     #         padding=1,
        #     #     #         output_padding=1,
        #     #     #         groups=num_classes
        #     #     #     )
        #     #     # )
        #     #
        #     #     up_tower.append(
        #     #         nn.Conv2d(
        #     #             num_classes,
        #     #             num_classes,
        #     #             3, 1, 1, groups=num_classes
        #     #         )
        #     #     )
        #     up_tower = [nn.UpsamplingBilinear2d(scale_factor=2**(int(math.log2(stride))-2)),
        #                 nn.Conv2d(num_classes, num_classes, 3, 1, 1, groups=num_classes)]
        #     up_towers.append(nn.Sequential(*up_tower))
        # self.up_towers = nn.ModuleList(up_towers)

        # init_modules = [self.cls_tower, self.cls_logits, self.gau_logits] if self.use_more_logits\
        init_modules = [self.cls_tower, self.cls_logits, self.bbox_tower, self.gau_logits] if self.use_more_logits\
            else [self.cls_tower, self.cls_logits]
        for modules in init_modules:  # , self.up_towers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                elif isinstance(l, nn.ConvTranspose2d):
                    torch.nn.init.constant_(l.weight, 1.0/9)
                    torch.nn.init.constant_(l.bias, 0)

        prior_prob = cfg.MODEL.GAU.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # for up_tower in self.up_towers:
        #     torch.nn.init.constant_(up_tower[-1].weight, 0)
        #     torch.nn.init.constant_(up_tower[-1].bias, bias_value)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.gau_logits.bias, bias_value)

    def forward(self, x):
        cls_logits, gau_logits = [], []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            # cls_logits = self.up_towers[l](cls_logits)
            cls_logits.append(self.cls_logits(cls_tower))
            if self.use_more_logits:
                bbox_tower = self.bbox_tower(feature)
                gau_logits.append(self.gau_logits(bbox_tower))
                # dif_logits.append(self.dif_logits(cls_tower))

        # H, W = logits[0].shape[-2:]
        # logits = [logit[:, :, :H, :W] for logit in logits]
        return cls_logits, gau_logits


class GauModule(torch.nn.Module):
    def __init__(self, cfg, in_channles):
        super(GauModule, self).__init__()
        self.head = GauHead(cfg, in_channles)

        self.box_selector_test = make_gau_postprocessor(cfg)
        self.loss_evaluator = make_gau_loss_evaluator(cfg)
        self.vis_labels = cfg.MODEL.GAU.DEBUG.VIS_LABELS
        self.vis_infer = cfg.MODEL.GAU.DEBUG.VIS_INFER
        self.dug_eval_gt = False

        self.loc_strides = cfg.MODEL.GAU.FPN_STRIDES  # [4] * 5

    def forward(self, images, features, targets=None):
        logits = self.head(features)
        cls_logits, gau_logits = logits
        locations = self.compute_locations(cls_logits, strides=self.loc_strides)

        if self.vis_labels:
            show_image(images, targets, self.loss_evaluator, self.box_selector_test, locations, self.loc_strides,
                       logits)

        if self.training:
            return self._forward_train(locations, logits, targets)
        else:
            if self.dug_eval_gt:  # test on ground-truth
                return eval_gt(self, locations, targets, images, logits)

            if self.vis_infer:
                return self._forward_test(self.loc_strides, logits, images.image_sizes, images, targets)
            else:
                return self._forward_test(self.loc_strides, logits, images.image_sizes)

    def _forward_train(self, locations, logits, targets):
        loss = self.loss_evaluator(locations, logits, targets)
        # loss = {
        #     "neg_loss": neg_loss,
        #     "pos_loss": pos_loss
        # }
        return None, loss

    def _forward_test(self, loc_strides, score_logits, image_sizes, images=None, targets=None):
        boxes = self.box_selector_test(loc_strides, score_logits, image_sizes, images, targets)
        return boxes, {}

    def compute_locations(self, cls_logits, strides):
        """
        iter each fpn level
        """
        fpn_locations = []
        for cls_logit, stride in zip(cls_logits, strides):
            H, W = cls_logit.shape[-2:]
            device = cls_logit.device
            shifts_x = torch.arange(
                0, W * stride, step=stride,
                dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, H * stride, step=stride,
                dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            locations = torch.stack((shift_x, shift_y), dim=1) + (stride - 1) / 2

            fpn_locations.append(locations)
        return fpn_locations


def build_gaussian_net(cfg, in_channels):
    return GauModule(cfg, in_channels)


def show_image(images, targets, loss_eval, infer, locations, loc_strides, logits):
    import numpy as np
    import matplotlib.pyplot as plt
    cls_logits, gau_logits = logits
    eps = 1e-6
    # vis label infer result
    labels, matched_gt_idxs = loss_eval.prepare_targets(locations, targets)
    labels_flatten = []
    gau_labels = []
    for l in range(len(labels)):
        N, C, H, W = cls_logits[l].shape
        labels_flatten.append(labels[l].reshape(N, H, W, -1).permute(0, 3, 1, 2))
        gau_labels.append((labels_flatten[l] > eps).float())
    infer.is_logit = False
    boxlists = infer(loc_strides, (labels_flatten, gau_labels), images.image_sizes)
    infer.is_logit = True
    infer_label = boxlists[0].bbox.cpu().numpy().tolist()
    # assert len(targets[0].bbox) == len(infer_label), "{} vs {}".format(len(targets[0].bbox), len(infer_label))
    # if len(targets[0].bbox) != len(infer_label):
    plt.figure()
    img = images.tensors[0].permute((1, 2, 0)).cpu().numpy() + np.array([102.9801, 115.9465, 122.7717])
    plt.imshow(img / 255)
    for (x1, y1, x2, y2) in targets[0].bbox.cpu().numpy().tolist():
        plt.axes().add_patch(
            plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, color=(1, 0, 0), linewidth=2)
        )
    for (ix1, iy1, ix2, iy2) in infer_label:
        plt.axes().add_patch(
            plt.Rectangle((ix1 + 5, iy1+5,), ix2 - ix1 + 1, iy2 - iy1 + 1, fill=False, color=(0, 1, 0), linewidth=2)
        )
    plt.title("image gt:{}, infer: {}".format(len(targets[0].bbox), len(infer_label)))
    plt.show()


def eval_gt(self, locations, targets, images, cls_logits):
    def eval_load_gt_bbox():
        #set TEST_FILTER_IGNORE: True, target will filter non-ignore out, else will keep valid ignore box.
        #if evaluate use 'ignore', set TEST_FILTER_IGNORE to True, else set to False
        [t.add_field("scores", torch.ones(len(t.bbox))) for t in targets]
        res = targets, {}
        return res

    def eval_target():
        labels_flatten = []
        for l in range(len(labels)):
            N, C, H, W = cls_logits[l].shape
            labels_flatten.append(labels[l].reshape(N, H, W, -1).permute(0, 3, 1, 2))
        self.box_selector_test.is_logit = False
        res = self._forward_test(self.loc_strides, labels_flatten, images.image_sizes)
        self.box_selector_test.is_logit = True
        return res

    def is_valid(valids_pos, l, b, y, x, c):
        valid_pos = valids_pos[l]
        # valid only for center, board have none, center
        B, H, W, C = valid_pos.shape
        if y == 0 or y == H + 2 - 1: return False
        elif x==0 or x == W + 2 - 1: return False
        return valid_pos[b, y-1, x-1, c]

    targets = [t.to(locations[0].device) for t in targets]

    labels, matched_gt_idxs = self.loss_evaluator.prepare_targets(locations, targets)
    # shape is fpn_level, (B, H, W, C)
    _, _, matched_gt_idxs = self.loss_evaluator.reshape(labels, cls_logits, matched_gt_idxs)
    valids_pos = self.loss_evaluator.valid_pos(matched_gt_idxs)

    # return eval_target()
    # return eval_load_gt_bbox()

    """
    get infer result of each point:
        bbox
        fpn_level, c, y, x
    get gt_idx of point by (fpn_level, c, y, x) in matched_gt_idxs
        gt_idx = matched_gt_idxs[fpn_level][b][y, x, c]
    if gt_idx >= 0 (valid), use target bbox inplace infer result
        bbox = targets[b][gt_idx]
    """
    boxlists, _ = self._forward_test(self.loc_strides, cls_logits, images.image_sizes)
    new_boxlists = []
    for b, boxlist in enumerate(boxlists):
        idx = boxlist.extra_fields['debug.feature_points']  # [[fpn_level, c, y, x]]
        for i, (l, c, y, x) in enumerate(idx):
            gt_idx = matched_gt_idxs[l][b][y, x, c]
            valid = is_valid(valids_pos, l, b, y, x, c)
            if gt_idx >= 0 and valid:
                boxlist.bbox[i, :] = targets[b].bbox[gt_idx]
            # filter neg box out
            else:
                boxlist.bbox[i, :] = -1
        boxlist = boxlist.clip_to_image(remove_empty=True)
        new_boxlists.append(boxlist)
    boxlists = new_boxlists

    boxlists = self.box_selector_test.select_over_all_levels(boxlists)
    # show_results(images, boxlists, targets)
    res = boxlists, {}
    return res


def show_results(images, boxlists, targets):
    from pycocotools.coco import COCO
    import matplotlib.pyplot as plt
    import numpy as np
    coco = COCO("/home/hui/dataset/voc/VOC2007/Annotations/pascal_test2007.json")

    det_results = torch.cat([boxlists[0].bbox, boxlists[0].extra_fields["labels"].reshape((-1, 1)).float(),
                             boxlists[0].extra_fields["scores"].reshape((-1, 1)),
                             boxlists[0].extra_fields["points"].float()], dim=1)
    det_results = det_results.cpu().numpy().tolist()
    img = images.tensors[0].permute((1, 2, 0)).cpu().numpy() + np.array([102.9801, 115.9465, 122.7717])
    plt.imshow(img / 255)
    for det in det_results:
        x1, y1, x2, y2, l, s, pc, py, px = det
        w, h = x2 - x1 + 1, y2 - y1 + 1
        x1, y1, w, h, l = [round(e) for e in [x1, y1, w, h, l]]
        if s < 0.3: continue
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='b', linewidth=2)
        plt.axes().add_patch(rect)
        plt.text(x1, y1, coco.loadCats(l)[0]['name'] + str(round(s, 2)))
        plt.scatter(px, py, color='r', s=20 * s)
        print('label', l, (x1, y1, w, h))

    if targets is not None:
        for (x1, y1, x2, y2) in targets[0].bbox.cpu().numpy().tolist():
            plt.axes().add_patch(
                plt.Rectangle((x1-2, y1-2), x2 - x1 + 1, y2 - y1 + 1, fill=False, color=(0, 1, 0), linewidth=2)
            )
    plt.show()
