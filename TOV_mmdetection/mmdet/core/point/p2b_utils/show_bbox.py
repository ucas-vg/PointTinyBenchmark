import torch
import cv2
import numpy as np


def show_box(self, proposals_list, filtered_scores, neg_proposal_list, neg_weight_list, bbox_results, gt_points,
             img_metas):

    for img in range(len(img_metas)):
        pos_box = proposals_list[img]

        # neg_box = neg_proposal_list[i]
        # neg_weight = neg_weight_list[i]
        gt_box = gt_points[img]
        img_meta = img_metas[img]
        filename = img_meta['filename']
        igs = cv2.imread(filename)
        h, w, _ = img_metas[img]['img_shape']
        igs = cv2.resize(igs, (w, h))
        import copy
        igs1 = copy.deepcopy(igs)
        boxes = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
        gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
        if filtered_scores:
            filtered_score = filtered_scores[img]
            cls_score = filtered_score['cls_score']
            ins_score = filtered_score['ins_score']
            dynamic_weight = filtered_score['dynamic_weight']

        for i in range(len(gt_box)):
            igs1 = cv2.rectangle(igs1, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                 color=(0, 255, 0))
            igs = cv2.rectangle(igs, (gt_box[i, 0], gt_box[i, 1]), (gt_box[i, 2], gt_box[i, 3]),
                                color=(0, 255, 0))
        for i in range(len(boxes)):
            color = (np.random.randint(0, 255), np.random.randint(0, 255),
                     np.random.randint(0, 255))
            igs1 = copy.deepcopy(igs)

            for j in range(len(boxes[i])):
                # if neg_weight[i]:

                blk = np.zeros(igs1.shape, np.uint8)
                blk = cv2.rectangle(blk, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                    color=color, thickness=-1)
                # 得到与原图形大小形同的形状

                igs1 = cv2.addWeighted(igs1, 1.0, blk, 0.3, 1, dst=None, dtype=None)
                igs1 = cv2.rectangle(igs1, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                     color=color, thickness=2)
                if filtered_scores:
                    igs1 = cv2.putText(igs1, str(cls_score[i, j]), (boxes[i, j, 0], boxes[i, j, 1]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                    cls = cls_score[i]
                    ins = ins_score[i]
                    dyna = dynamic_weight[i]

                # cv2.imwrite('exp/debug/'+filename,igs1)
                cv2.namedWindow("ims1", 0)
                cv2.resizeWindow("ims1", 2000, 1200)
                cv2.imshow('ims1', igs1)
                # cv2.namedWindow("ims", 0)
                # cv2.resizeWindow("ims", 1333, 800)
                # cv2.imshow('ims', igs)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                elif cv2.waitKey(0) & 0xFF == ord('b'):
                    break

    def show_imgs(self, img_metas, gt_labels, pseudo_boxes,
                  gt_bboxes):
        import cv2
        for i in range(len(img_metas)):
            gt_box = gt_bboxes[i]
            pos_box = pseudo_boxes[i]

            img_meta = img_metas[i]

            pos_box = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
            gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
            ims = cv2.imread(img_meta['filename'])
            im_h, im_w, _ = img_meta['img_shape']
            ims = cv2.resize(ims, (im_w, im_h))
            for k in range(len(gt_labels[i])):
                cls_scores
                ims = cv2.rectangle(ims, (gt_box[k, 0], gt_box[k, 1]), (gt_box[k, 2], gt_box[k, 3]),
                                    color=(0, 255, 0))
            # for i in range(len(gt_labels[i])):
            #     ims = cv2.rectangle(ims, (pos_box[i, 0], pos_box[i, 1]), (pos_box[i, 2], pos_box[i, 3]),
            #                          color=(0, 255, 0))

            for j in range(len(gt_labels[i])):
                m = min(gt_inds_)
                heatmap = cls_scores_[gt_inds_ - m == j]
                if len(heatmap) > 0:
                    heatmap = heatmap.sigmoid()
                    heatmap = heatmap.mean(dim=0).squeeze(0)
                    # heatmap = heatmap.permute(1, 0)
                    # heatmap[heatmap>0.1]=1
                    heatmap = np.array(heatmap.cpu().detach())
                    heatmapshow = None
                    # heatmap[0, 0] = 1
                    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)

                    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                    pad_h, pad_w, _ = img_meta['pad_shape']
                    im_h, im_w, _ = img_meta['img_shape']
                    heatmapshow = cv2.resize(heatmapshow, (w * 4, h * 4))

                    heatmapshow = heatmapshow[:im_h, :im_w, :]
                    img = cv2.imread(img_meta['filename'])
                    img = cv2.resize(img, (im_w, im_h))
                    img = cv2.rectangle(img, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                        color=(0, 255, 0))
                    img = cv2.rectangle(img, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                        color=(0, 255, 255))
                    heatmapshow = cv2.rectangle(heatmapshow, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                                color=(0, 255, 0))
                    heatmapshow = cv2.rectangle(heatmapshow, (pos_box[j, 0], pos_box[j, 1]),
                                                (pos_box[j, 2], pos_box[j, 3]),
                                                color=(0, 255, 255))

                    cv2.namedWindow("ims1", 0)
                    cv2.resizeWindow("ims1", 640, 480)
                    cv2.imshow('ims1', ims)
                    cv2.namedWindow("ht", 0)
                    cv2.resizeWindow("ht", im_w * 4, im_h * 4)
                    cv2.imshow('ht', heatmapshow)
                    cv2.namedWindow("im", 0)
                    cv2.resizeWindow("im", im_w * 4, im_h * 4)
                    cv2.imshow('im', img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

