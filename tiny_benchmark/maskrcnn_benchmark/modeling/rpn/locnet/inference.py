import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        debug_vis_label=False,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.debug_vis_label = debug_vis_label

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        if centerness is not None:
            centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
            centerness = centerness.reshape(N, -1).sigmoid()
            box_cls = box_cls * centerness[:, :, None]

        if self.debug_vis_label:
            # box_prob_set.extend([box_cls, centerness, centerness[:,:,None]*box_prob_set[-1]])
            show_box_cls([box_cls, box_cls**2], N, H, W, C, self.pre_nms_thresh)

        # K = 1
        # box_cls = box_cls.reshape(-1, C)
        # top, idim = torch.topk(box_cls, K, dim=-1)
        # box_cls[:] = 0
        # i0 = torch.zeros(idim.size()).long() + torch.arange(0, idim.size(0))[:, None]
        # box_cls[i0, idim] = top
        # box_cls = box_cls.reshape(N, -1, C)

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            if self.debug_vis_label: boxlist.add_field("det_locations", per_locations)  # add by hui
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes, images=None, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        if centerness is None:
            centerness = [None] * len(box_cls)
        if self.debug_vis_label:
            show_box_cls.images = images
            show_box_cls.targets = targets

        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            if self.debug_vis_label:
                locations = boxlists[i].get_field("det_locations")  # add here
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                if self.debug_vis_label:
                    locations_j = locations[inds]
                    boxlist_for_class.add_field("det_locations", locations_j)  # add here
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_location_postprocessor(config):
    pre_nms_thresh = config.MODEL.LOC.INFERENCE_TH
    pre_nms_top_n = config.MODEL.LOC.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.LOC.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.LOC.NUM_CLASSES,
        debug_vis_label=config.MODEL.LOC.DEBUG.VIS_LABELS
    )

    return box_selector


import torch.nn.functional as F
import numpy as np
class BoxClsShower(object):
    """
    map0-4: [0.125, 0.25, 0.5, 0.75, 1.0] ** 2 area range
    map5: centerness
    """
    def __init__(self, fpn_level=5, scatter_topk=10, EPS=1e-8, images=None, targets=None):
        self.fpn_level = fpn_level
        self.level_count = 0
        self.box_probs = []
        self.scatter_topk = scatter_topk
        self.EPS = EPS
        self.row_sub_fig = 4
        self.single_fig_size = 4
        self.titles = None
        self.images = images
        self.targets = targets

    def find_local_max(self, box_prob):
        B, C, H, W = box_prob.shape
        max_prob, idx = F.max_pool2d_with_indices(box_prob, 3, 1, 1, return_indices=True)
        max_prob = max_prob[0, 0]
        box_prob = box_prob[0, 0]
        is_local_max = torch.nonzero(box_prob == max_prob)
        y, x = is_local_max[:, 0], is_local_max[:, 1]
        idx = torch.argsort(-box_prob[y, x])
        k = self.scatter_topk
        y = y[idx[:k]]
        x = x[idx[:k]]
        return y.cpu().numpy(), x.cpu().numpy(), box_prob[y, x]

    def mask_to_image(self, box_prob, upsample=False):
        img = self.images.tensors[0:1]
        if upsample:
            box_prob = F.upsample(box_prob[None, None, :, :], img.shape[2:], mode='bilinear')[0, 0]
        else:
            img = F.upsample(img, box_prob.shape, mode='bilinear')
        img = img[0].permute((1, 2, 0)).cpu() + torch.Tensor([102.9801, 115.9465, 122.7717])
        return img * box_prob[:, :, None] / 255

    def __call__(self, box_prob_set, N, H, W, C, th):
        import matplotlib.pyplot as plt
        box_probs = []
        for i, box_prob in enumerate(box_prob_set):
            if box_prob.numel() == N*H*W*C:
                box_prob = box_prob.reshape(-1, H, W, C)[:1]
            elif box_prob.numel() == N*H*W:
                box_prob = box_prob.reshape(-1, H, W, 1)[:1]
            box_prob = box_prob.max(dim=-1, keepdim=True)[0].permute((0, 3, 1, 2))
            box_probs.append(box_prob.detach().cpu())

        # merge FPN multi-level score map to one map by resize add.
        if len(self.box_probs) == 0:
            self.box_probs = box_probs
        else:
            for i, p in enumerate(box_probs):  # for each area th score map
                box_prob = self.box_probs[i]
                if box_prob.numel() < p.numel():
                    box_prob = F.upsample(box_prob, p.shape[2:], mode='bilinear')
                else:
                    p = F.upsample(p, box_prob.shape[2:], mode='bilinear')
                self.box_probs[i] = torch.max(torch.stack([p, box_prob]), dim=0)[0]
        self.level_count += 1

        if self.level_count == self.fpn_level:
            # show each area th score map
            n_figs = len(self.box_probs)
            r = self.row_sub_fig if n_figs >= self.row_sub_fig else n_figs
            c = int(np.ceil((n_figs/r)))
            plt.figure(figsize=(r * self.single_fig_size, c * self.single_fig_size))  # (W, H)
            for i, box_prob in enumerate(self.box_probs):
                y, x, score = self.find_local_max(box_prob)
                box_prob = box_prob[0, 0]
                max_p = box_prob.max()
                std = box_prob.std()
                box_prob /= max_p
                if self.images is not None:
                    box_prob = self.mask_to_image(box_prob)
                box_prob = box_prob.numpy()
                plt.subplot(c, r, i+1)
                plt.imshow(box_prob)
                plt.scatter(x, y, color='r', s=20 * score)
                if self.titles is not None:
                    title = self.titles[i]
                else:
                    title = 'map {}'.format(i)
                plt.title("{}, max:{:.2f}, std: {:.2f}".format(title, float(max_p), float(std)))
            plt.show()
            self.level_count = 0
            del self.box_probs
            self.box_probs = []


show_box_cls = BoxClsShower()
