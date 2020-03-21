import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
import torch.nn.functional as F
from .gau_label_infer import cross_points_set_solve_3d


class GAUPostProcessor(torch.nn.Module):
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
        super(GAUPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        self.local_max_kernel = 3
        assert self.local_max_kernel % 2 == 1, "local max kernel must odd."
        self.local_max_pad = (self.local_max_kernel - 1) // 2
        self.max_top_k = 500

        self.infer_a = 1
        self.infer_b = 1
        beta = 2
        inflection_point = 0.25
        self.sigma = inflection_point * ((beta / (beta - 1)) ** (1.0 / beta))
        self.is_logit = True

    def find_local_max(self, box_prob_per_im, gau_prob_per_im):
        C, H, W = gau_prob_per_im.shape
        max_prob, _ = F.max_pool2d_with_indices(
            gau_prob_per_im.unsqueeze(dim=0), self.local_max_kernel, 1, self.local_max_pad)  # , return_indices=True)
        # is_local_max = torch.nonzero((box_prob_per_im == max_prob[0])  # local max filter
        #                              & (box_prob_per_im > self.pre_nms_thresh))  # threshold filter

        is_local_max = torch.nonzero(
            ((gau_prob_per_im == max_prob[0]) & (box_prob_per_im > self.pre_nms_thresh))  # threshold filter
            # | (box_prob_per_im > 0.4)
        )

        # remove boarder point
        c, y, x = is_local_max[:, 0], is_local_max[:, 1], is_local_max[:, 2]
        no_board = (y > 0) & (y < H - 1) & (x > 0) & (x < W - 1)
        is_local_max = is_local_max[no_board]
        c, y, x = is_local_max[:, 0], is_local_max[:, 1], is_local_max[:, 2]
        gau_prob_per_im = gau_prob_per_im[c, y, x]  # attention: not gau_prob_per_im[is_local_max]
        box_prob_per_im = box_prob_per_im[c, y, x]

        # top-k filter
        if len(gau_prob_per_im) > self.pre_nms_top_n:
            prob_per_im, top_idx = torch.topk((gau_prob_per_im * box_prob_per_im) ** 0.5, self.pre_nms_top_n)
            is_local_max = is_local_max[top_idx]
        else:
            prob_per_im = (gau_prob_per_im * box_prob_per_im) ** 0.5
        return is_local_max, prob_per_im

    def forward_for_single_feature_map(
            self, step, box_cls, gau_logits,
            image_sizes, level):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        if self.is_logit:
            box_prob = box_cls.sigmoid()
            gau_prob = gau_logits.sigmoid()
        else:
            box_prob = box_cls
            gau_prob = gau_logits
        L = -torch.log(gau_prob) * self.sigma * self.sigma

        # put in the same format as locations
        # box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        # box_cls = box_cls.reshape(N, -1, C).sigmoid()
        # box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        # box_regression = box_regression.reshape(N, -1, 4)
        # centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()

        # pre nms filter
        # N, C, H, W = box_prob.shape
        # candidate_inds = box_prob > self.pre_nms_thresh
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # boxes = cross_points_set_solve_3d(L, local_max_point, self.infer_a, self.infer_b)
        results = []
        for i in range(N):
            # top-k + threshold filter
            # per_box_cls = box_cls[i]
            # per_candidate_inds = candidate_inds[i]
            # per_box_cls = per_box_cls[per_candidate_inds]
            #
            # per_candidate_nonzeros = per_candidate_inds.nonzero()
            # per_box_loc = per_candidate_nonzeros[:, 0]
            # per_class = per_candidate_nonzeros[:, 1] + 1
            #
            # per_locations = locations[per_box_loc]
            #
            # per_pre_nms_top_n = pre_nms_top_n[i]
            #
            # if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
            #     per_box_cls, top_k_indices = \
            #         per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            #     per_class = per_class[top_k_indices]
            #     per_box_regression = per_box_regression[top_k_indices]
            #     per_locations = per_locations[top_k_indices]

            # print(box_prob[i].max())
            local_max_point, per_box_cls = self.find_local_max(box_prob[i], gau_prob[i])
            # print(len(local_max_point))
            if len(local_max_point) > 0:
                bboxes1 = cross_points_set_solve_3d(L[i], local_max_point, self.infer_a, self.infer_b, step=step, solver=1)
                bboxes2 = cross_points_set_solve_3d(L[i], local_max_point, self.infer_a, self.infer_b, step=step, solver=2)
                eps = 1e-6
                err = ((bboxes1[:, :4] + eps) / (bboxes2[:, :4] + eps)).min(dim=1)[0]
                bboxes1[(err > 1.2) & (err < 1/1.2), :] = 0
                bboxes = bboxes1
                bboxes[:, :2] += (step - 1) / 2  # cause compute location +(step-1)/2
            else:
                bboxes = torch.FloatTensor(size=(0, 5)).to(local_max_point.device)
            w, h = bboxes[:, 2], bboxes[:, 3]
            keep = (w > 0) & (h > 0)
            bboxes = bboxes[keep]
            local_max_point = local_max_point[keep]
            x1, y1, w, h = bboxes[:, :4].transpose(0, 1)
            detections = torch.stack([
                x1, y1, x1 + w - 1, y1 + h - 1
            ], dim=1)
            # detections = torch.FloatTensor(size=(0, 4)).to(bboxes.device)
            per_class = local_max_point[:, 0] + 1

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            # add for debug and vis
            boxlist.add_field("points", local_max_point * step + (step - 1) / 2)
            fpn_level = torch.ones(len(local_max_point), 1).to(local_max_point.device)*level
            boxlist.add_field("debug.feature_points", torch.cat([fpn_level.long(), local_max_point], dim=1))
            # #######################################################################
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def forward(self, loc_strides, logits, image_sizes, images=None, targets=None):
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
        sampled_boxes = []
        cls_logits, gau_logits = logits
        for level, (s, c, gl) in enumerate(zip(loc_strides, cls_logits, gau_logits)):
            bboxes = self.forward_for_single_feature_map(s, c, gl, image_sizes, level)
            if len(bboxes) > 0:
                sampled_boxes.append(bboxes)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        if images is not None:
            show_images(images, boxlists, targets)
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
                boxlist_for_class.add_field("points", boxlists[i].get_field("points")[inds])  # add here
                boxlist_for_class.add_field("debug.feature_points", boxlists[i].get_field("debug.feature_points")[inds])  # add here
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


def make_gau_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = GAUPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES
    )

    return box_selector


def show_images(images, boxlists, targets):
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

    if targets is not None:
        for (x1, y1, x2, y2) in targets[0].bbox.cpu().numpy().tolist():
            plt.axes().add_patch(
                plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, color=(0, 1, 0), linewidth=2)
            )
    plt.show()

# iter = 0
# def show_label_map(box_cls):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     sigmoid = lambda x: 1 / (1 + np.exp(-x))
#     global iter
#     iter += 1
#     if iter % 20 != 0: return
#     new_clses = []
#     for i, cls in enumerate(box_cls):
#         new_clses.append(cls.cpu().detach().numpy().transpose((0, 2, 3, 1)))
#     box_cls = new_clses
#
#     N = sum([(label[0].sum(axis=(0, 1)) > 0).sum() for label in box_cls])
#     n = 1
#     print(N)
#     # plt.figure(figsize=())
#     for l, label in enumerate(box_cls):
#         for c in range(label.shape[-1]):
#             if label[0, :, :, c].sum() > 0:
#                 plt.subplot(N, 2, n)
#                 print(label.shape)
#                 plt.imshow(np.log(label[0, :, :, c] + 0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal
#                 plt_str = ("pos_count:{}; {}, {}".format((label[0, :, :, c] > 0).sum(), l, c))
#                 plt.title(plt_str)
#                 plt.subplot(N, 2, n + 1)
#                 pred = box_cls[l][0, :, :, c]
#                 plt.imshow(np.log(pred+0.01), vmin=np.log(0.01), vmax=np.log(1.01))  # if no vmin vmax set, will linear normal, not show absolute value
#                 plt.title("s: [{:.4f}, {:.4f}]".format(
#                     sigmoid(pred.min()), sigmoid(pred.max())))
#                 print(plt_str)
#                 n += 2
#     # plt.show()
#     plt.savefig("outputs/pascal/gau/base/iter_png/{}.png".format(iter))
