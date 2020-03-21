import torch
import torch.nn.functional as F


class OHEMLoss(object):
    def __init__(self, neg_rate, binary_logits=False):
        self.neg_rate = neg_rate
        self.cross_entropy = F.binary_cross_entropy_with_logits if binary_logits else F.cross_entropy
        self.sample_count = None

    def get_sample_count(self):
        return self.sample_count

    def __call__(self, pred, label):
        # pred = pred.permute(0, 1, 3, 4, 2)
        pos_flag = label > 0
        neg_flag = label == 0
        pos_label = label[pos_flag]
        neg_label = label[neg_flag]
        pos_pred = pred[pos_flag]
        neg_pred = pred[neg_flag]

        num_pos = len(pos_pred)
        if num_pos == 0:
            loss = torch.Tensor([0.]).to(pred.device)
            loss.requires_grad = True
            return loss

        pos_loss = self.cross_entropy(pos_pred, pos_label, size_average=False, reduce=False)
        neg_loss = self.cross_entropy(neg_pred, neg_label, size_average=False, reduce=False)

        num_neg = num_pos * self.neg_rate if len(neg_pred) > num_pos * self.neg_rate else len(neg_pred)
        num_neg = int(num_neg)
        # _, idx = torch.sort(neg_loss, descending=True)
        # neg_loss = neg_loss[idx[:num_neg].long()]
        _, neg_idx = torch.topk(neg_loss, num_neg)
        neg_loss = neg_loss[neg_idx]
        # / (3 + 1)
        self.sample_count = len(pos_loss) + len(neg_loss)
        return (pos_loss.sum() + neg_loss.sum()) / self.sample_count


class OHEM2Loss(object):
    def __init__(self, batch_size, fg_fraction, binary_logits=False, hard_rate=1.0):
        self.batch_size = batch_size
        self.fg_frac = fg_fraction
        self.max_num_pos = self.batch_size * self.fg_frac
        self.cross_entropy = F.binary_cross_entropy_with_logits if binary_logits else F.cross_entropy
        self.hard_rate = hard_rate
        self.sample_count = None

    def get_sample_count(self):
        return self.sample_count

    def __call__(self, pred, label):
        pos_flag = label > 0
        neg_flag = label == 0
        pos_label = label[pos_flag]
        neg_label = label[neg_flag]
        pos_pred = pred[pos_flag]
        neg_pred = pred[neg_flag]

        if len(pos_pred) > 0:
            pos_loss = self.cross_entropy(pos_pred, pos_label, size_average=False, reduce=False)
            if len(pos_loss) > self.max_num_pos:
                idx = torch.randperm(pos_loss.size(0))
                pos_loss = pos_loss[idx[:int(self.max_num_pos)].long()]
            max_num_neg = self.batch_size - len(pos_loss)
        else:
            pos_loss = torch.Tensor([0.]).to(pred.device)
            pos_loss.requires_grad = True
            max_num_neg = self.batch_size

        neg_loss = self.cross_entropy(neg_pred, neg_label, size_average=False, reduce=False)
        if len(neg_loss) > max_num_neg:
            _, sort_idx = torch.sort(neg_loss, descending=True)
            num_hard = int(self.hard_rate * max_num_neg)
            num_random = int(max_num_neg - num_hard)
            if num_hard > 0: hard_neg_loss = neg_loss[sort_idx[:num_hard].long()]
            if num_random > 0:
                ridx = (torch.randperm(int(len(neg_loss) - num_hard)) + num_hard)[:num_random]
                rand_neg_loss = neg_loss[sort_idx[ridx.long()].long()]
            if num_hard == 0: neg_loss = rand_neg_loss
            elif num_random == 0: neg_loss = hard_neg_loss
            else: neg_loss = torch.cat((hard_neg_loss, rand_neg_loss), 0)
#         print(num_hard, num_random, max_num_neg, len(pos_loss))

        # / (3 + 1)
        self.sample_count = len(pos_loss) + len(neg_loss)
        return (pos_loss.sum() + neg_loss.sum()) / self.sample_count
