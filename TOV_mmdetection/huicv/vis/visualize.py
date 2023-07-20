import numpy as np
import matplotlib.pyplot as plt
import warnings
from .bbox_utils import *


def draw_a_bbox(box, color, linewidth=1, dash=False, fill=False, ax=None):
    if ax is None:
        ax = plt.gca()
    if dash:
        # box_to_dashed_rect(plt, box, color, linewidth)
        ax.add_patch(box_to_rect(box, color, linewidth, fill=fill, ls='--'))
    else:
        ax.add_patch(box_to_rect(box, color, linewidth, fill=fill))


def box_to_dashed_rect(box, color, linewidth=1, ax=None):
    if ax is None:
        ax = plt.gca()
    x1, y1, x2, y2 = box
    ax.plot([x1, x2], [y1, y1], '--', color=color, linewidth=linewidth)
    ax.plot([x1, x2], [y2, y2], '--', color=color, linewidth=linewidth)
    ax.plot([x1, x1], [y1, y2], '--', color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y1, y2], '--', color=color, linewidth=linewidth)


def box_to_rect(box, color, linewidth=1, fill=False, ls='-'):
    """convert an anchor box to a matplotlib rectangle"""
    alpha = 0.2 if fill else None
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                  fill=fill, alpha=alpha, edgecolor=color, facecolor=color,
                  linewidth=linewidth, linestyle=ls)


def draw_bbox(fig, bboxes, color=(0, 0, 0), linewidth=1, fontsize=5, normalized_label=True, wh=None,
              show_text=False, class_names=None, class_colors=None, use_real_line=None, threshold=None):
    """
        draw boxes on fig
    argumnet:
        bboxes: [[x1, y1, x2, y2, (cid), (score) ...]],
        color: box color, if class_colors not None, it will not use.
        normalized_label: if label xmin, xmax, ymin, ymax is normaled to 0~1, set it to True and wh must given, else set to False.
        wh: (image width, height) needed when normalized_label set to True
        show_text: if boxes have cid or (cid, score) dim, can set to True to visualize it.
        use_real_line: None means all box use real line to draw, or [boolean...] means whether use real line for per class label.
        class_names: class name for every class.
        class_colors: class gt box color for every class, if set, argument 'color' will not use
    """
    if len(bboxes) == 0: return
    if np.max(bboxes) <= 1.:
        if normalized_label == False: warnings.warn(
            "[draw_bbox]:the label boxes' max value less than 1.0, may be it is noramlized box," +
            "maybe you need set normalized_label==True and specified wh", UserWarning)
    else:
        if normalized_label == True: warnings.warn(
            "[draw_bbox]:the label boxes' max value bigger than 1.0, may be it isn't noramlized box," +
            "maybe you need set normalized_label==False.", UserWarning)

    if normalized_label:
        assert wh != None, "wh must be specified when normalized_label is True. maybe you need setnormalized_label=False "
        bboxes = inv_normalize_box(bboxes, wh[0], wh[1])

    if color is not None and class_colors is not None:
        warnings.warn("'class_colors' set, then 'color' will not use, please set it to None")

    for box in bboxes:
        # [x1, y1, x2, y2, (cid), (score) ...]
        if len(box) >= 5 and box[4] < 0: continue  # have cid or not
        if len(box) >= 6 and threshold is not None and box[5] < threshold: continue
        if len(box) >= 5 and class_colors is not None: color = class_colors[int(box[4])]
        if len(box) >= 5 and use_real_line is not None and not use_real_line[int(box[4])]:
            box_to_dashed_rect(box[:4], color, linewidth, ax=fig)
        else:
            rect = box_to_rect(box[:4], color, linewidth)
            fig.add_patch(rect)
        if show_text:
            cid = int(box[4])
            if class_names is not None: cid = class_names[cid]
            text = str(cid)
            if len(box) >= 6: text += " {:.3f}".format(box[5])
            fig.text(box[0], box[1], text,
                     bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))


def draw_center(ax, bboxes, **kwargs):
    center = np.array([((x1+x2)/2, (y1+y2)/2)for x1, y1, x2, y2, *arg in bboxes])
    ax.scatter(center[:, 0], center[:, 1], **kwargs)


def get_hsv_colors(n):
    import colorsys
    rgbs = [None] * n
    for i in range(n):
        h = i / n
        rgbs[i] = colorsys.hsv_to_rgb(h, 1, 1)
    return rgbs
