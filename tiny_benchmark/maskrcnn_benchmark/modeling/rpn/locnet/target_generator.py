import torch
from math import sqrt


INF = 100000000


def one_hot(label, num_classes):
    class_range = torch.arange(1, num_classes+1, dtype=label.dtype, device=label.device).unsqueeze(0)
    label = label.reshape((-1, 1))
    return label == class_range


class FCOSTarget(object):
    def __init__(self, num_classes, cls_pos_area=1.0, only_reg=False):
        self.num_classes = num_classes
        self.only_reg = only_reg
        self.cls_pos_area = cls_pos_area
        self.no_match_gt_count = 0  # add by hui

    def __call__(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            reg_targets.append(reg_targets_per_im)

            # cls label
            if not self.only_reg:   # add by hui
                if self.cls_pos_area < 1:
                    is_in_boxes = self.is_in_pos_boxes(xs, ys, targets_per_im, self.cls_pos_area)
                    locations_to_gt_area = area[None].repeat(len(locations), 1)
                    locations_to_gt_area[is_in_boxes == 0] = INF
                    locations_to_gt_area[is_cared_in_the_level == 0] = INF
                    locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

                labels_per_im = targets_per_im.get_field("labels")
                labels_per_im = labels_per_im[locations_to_gt_inds]
                labels_per_im[locations_to_min_aera == INF] = 0

                labels_per_im = one_hot(labels_per_im, self.num_classes).float()  # add
                labels.append(labels_per_im)

        return labels, reg_targets

    # add by hui for area label
    def is_in_pos_boxes(self, xs, ys, targets_per_im, pos_area, EPS=1e-6):
        bboxes = targets_per_im.bbox
        centers = torch.cat([(bboxes[:, [0]] + bboxes[:, [2]]) / 2, (bboxes[:, [1]] + bboxes[:, [3]]) / 2], dim=1)
        WH = torch.cat([(bboxes[:, [2]] - bboxes[:, [0]] + 1), (bboxes[:, [3]] - bboxes[:, [1]] + 1)], dim=1)
        WH *= sqrt(pos_area)
        # WH[WH < 2 + EPS] = 2 + EPS
        x1y1 = centers - (WH - 1) / 2
        x2y2 = centers + (WH - 1) / 2
        bboxes = torch.cat([x1y1, x2y2], dim=1)

        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        ltrb = torch.stack([l, t, r, b], dim=2)
        is_in_boxes = ltrb.min(dim=2)[0] > 0

        self.no_match_gt_count += (is_in_boxes.sum(dim=0) < EPS).sum().item()
        if (self.no_match_gt_count + 1) % 100 == 0:
            import warnings
            warnings.warn("when pos_area={}, already have {} ground-truth no matched."
                          .format(pos_area, self.no_match_gt_count))
        return is_in_boxes


class FunctionTarget(object):
    def __init__(self, num_classes, *args, **kwargs):
        self.num_classes = num_classes

    def __call__(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            targets_per_im = targets_per_im.to(xs.device)
            bboxes = targets_per_im.bbox
            area = targets_per_im.area()

            # 1. match policy
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with maxmal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            # 2. get reg label
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            reg_targets.append(reg_targets_per_im)

            # 3. get cls label
            cls_label = self.cls_target_function(xs, ys, targets_per_im,
                                                 locations_to_gt_inds, locations_to_min_aera, reg_targets_per_im)
            labels.append(cls_label)
        return labels, reg_targets

    def cls_target_function(self, xs, ys, targets_per_im,
                            locations_to_gt_inds, locations_to_min_area, reg_targets_per_im):
        """
        :param xs: xs[location_id] x position in origin image
        :param ys: ys[location_id] y position in origin image
        :param targets_per_im: BoxList, bboxes[box_id, 4] gt, labels, areas
        :param reg_targets: FCOS regressin targets
        :return:
        """
        return NotImplementedError


class GaussianTarget(FunctionTarget):
    def __init__(self, num_classes, beta):
        super(GaussianTarget, self).__init__(num_classes)
        self.beta = beta
        self.inflection_point = 0.25

    def cls_target_function(self, xs, ys, targets_per_im, locations_to_gt_inds, locations_to_min_area, *args):
        beta = self.beta
        sigma = self.inflection_point * ((beta / (beta - 1)) ** (1.0/beta))

        bboxes = targets_per_im.bbox

        W = bboxes[:, 2] - bboxes[:, 0] + 1
        H = bboxes[:, 3] - bboxes[:, 1] + 1
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        # D = (((xs[:, None] - cx[None, :]).abs() / (sigma * W)) ** 2
        #      + ((ys[:, None] - cy[None, :]).abs() / (sigma * H)) ** 2)
        # Q = torch.exp(-D)

        l2g = locations_to_gt_inds
        D = ((xs - cx[l2g]).abs() / (sigma * W[l2g])) ** beta + ((ys - cy[l2g]).abs() / (sigma * H[l2g])) ** beta
        Q = torch.exp(-D)

        labels_per_im = targets_per_im.get_field("labels").to(xs.device)
        cls_label = torch.zeros(size=(len(xs), self.num_classes), device=xs.device)
        locations_to_class = labels_per_im[locations_to_gt_inds]
        cls_label[range(len(xs)), locations_to_class-1] = Q * (locations_to_min_area < INF).float()
        return cls_label


class CenterTarget(FunctionTarget):
    def __init__(self, num_classes, beta):
        super(CenterTarget, self).__init__(num_classes)
        self.beta = beta

    def cls_target_function(self, xs, ys, targets_per_im, locations_to_gt_inds, locations_to_min_area, reg_targets_per_im):
        labels_per_im = targets_per_im.get_field("labels")
        labels_per_im = labels_per_im[locations_to_gt_inds]
        labels_per_im[locations_to_min_area == INF] = 0
        labels = one_hot(labels_per_im, self.num_classes).float()  # add

        pos_idx = torch.nonzero(locations_to_min_area < INF)[:, 0]
        if len(pos_idx) > 0:
            pos_centerness = self.compute_centerness_targets(reg_targets_per_im[pos_idx])
            labels[pos_idx, labels_per_im[pos_idx] - 1] = pos_centerness ** self.beta
        return labels

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


class LOCTargetGenerator(object):
    def __init__(self, cfg):
        target_generator = cfg.MODEL.LOC.TARGET_GENERATOR
        num_cls = cfg.MODEL.LOC.NUM_CLASSES - 1
        if target_generator == 'gaussian':
            self.compute_targets_for_locations = GaussianTarget(num_cls, cfg.MODEL.LOC.LABEL_BETA)
        elif target_generator == 'centerness':
            self.compute_targets_for_locations = CenterTarget(num_cls, cfg.MODEL.LOC.LABEL_BETA)
        elif target_generator == 'fcos':
            self.compute_targets_for_locations = FCOSTarget(num_cls, cfg.MODEL.LOC.FCOS_CLS_POS_AREA)
        else:
            raise ValueError("cfg.MODEL.LOC.TARGET_GENERATOR {}, must in "
                             "['gaussian', 'fcos', 'centerness']".format(target_generator))

    def __call__(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


def build_target_generator(cfg):
    return LOCTargetGenerator(cfg)
