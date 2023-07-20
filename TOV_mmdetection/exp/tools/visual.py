import cv2
import torch
import numpy as np
import torch

def show_imgs(cls_scores, img_metas, gt_labels, pseudo_boxes, gt_bboxes):
    img_len =[len(i) for i in gt_labels]
    cls_scores=cls_scores.softmax(dim=1)
    cls_scores = cls_scores.split(img_len)

    for i in range(len(img_metas)):
        cls_scores_=cls_scores[i]
        gt_label = gt_labels[i]
        gt_box = gt_bboxes[i]
        pos_box = pseudo_boxes[i]
        img_meta = img_metas[i]
        cls_scores_=cls_scores_[torch.arange(len(gt_label)),gt_label,...]

        pos_box = np.array(torch.tensor(pos_box).cpu()).astype(np.int32)
        gt_box = np.array(torch.tensor(gt_box).cpu()).astype(np.int32)
        ims = cv2.imread(img_meta['filename'])
        im_h, im_w, _ = img_meta['img_shape']
        ims = cv2.resize(ims, (im_w, im_h))
        for k in range(len(gt_labels[i])):
            ims = cv2.rectangle(ims, (gt_box[k, 0], gt_box[k, 1]), (gt_box[k, 2], gt_box[k, 3]),
                                color=(0, 255, 0))
        # for i in range(len(gt_labels[i])):
        #     ims = cv2.rectangle(ims, (pos_box[i, 0], pos_box[i, 1]), (pos_box[i, 2], pos_box[i, 3]),
        #                          color=(0, 255, 0))

        for j in range(len(gt_labels[i])):
            heatmap = cls_scores_[j]
            # heatmap = heatmap.permute(1, 0)
            # heatmap[heatmap>0.1]=1
            heatmap = np.array(heatmap.cpu().detach())
            heatmapshow = None
            heatmap[0, 0] = 1
            heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U)

            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            pad_h, pad_w, _ = img_meta['pad_shape']
            im_h, im_w, _ = img_meta['img_shape']
            heatmapshow = cv2.resize(heatmapshow, (pos_box[j,2]-pos_box[j,0],pos_box[j,3]-pos_box[j,1]))
            # heatmapshow = heatmapshow[:im_h, :im_w, :]
            img = cv2.imread(img_meta['filename'])
            img = cv2.resize(img, (im_w, im_h))
            img = cv2.rectangle(img, (gt_box[j, 0], gt_box[j, 1]), (gt_box[j, 2], gt_box[j, 3]),
                                color=(0, 255, 0))
            img = cv2.rectangle(img, (pos_box[j, 0], pos_box[j, 1]), (pos_box[j, 2], pos_box[j, 3]),
                                color=(0, 255, 255))

            cv2.namedWindow("ims1", 0)
            cv2.resizeWindow("ims1", 640, 480)
            cv2.imshow('ims1', ims)
            cv2.namedWindow("ht", 0)
            cv2.resizeWindow("ht", heatmapshow.shape[1] * 10, heatmapshow.shape[0] * 10)
            cv2.imshow('ht', heatmapshow)
            cv2.namedWindow("im", 0)
            cv2.resizeWindow("im", im_w * 4, im_h * 4)
            cv2.imshow('im', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
