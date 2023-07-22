import torch
import json


def keep_center_replace_wh(B, WH):
    W, H = WH[:, 0], WH[:, 1]
    xc, yc = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
    x1, x2 = xc - W / 2, xc + W / 2
    y1, y2 = yc - H / 2, yc + H / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


class NMSStub(object):
    def __init__(self):
        # python exp/tools/cal_dataset_statistic.py
        label_mean_size = json.load(open('exp/tinycocotrain_label_mean_wh.json'))['label2mean_size']
        self.label_mean_size = torch.tensor(label_mean_size)
        print("[NMSStub]")

    def __call__(self, bboxes, labels):
        self.label_mean_size = self.label_mean_size.to(bboxes)
        bboxes = keep_center_replace_wh(bboxes, self.label_mean_size[labels])
        return bboxes


nms_stub = NMSStub()
