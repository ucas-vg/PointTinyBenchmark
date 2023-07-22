import torch.nn.functional as F
from collections import defaultdict
import torch
from mmdet.utils.logger import get_root_logger
import numpy as np


def swap_list_order(alist):
    """
    Args:
        alist: shape=(B, num_level, ....)
    Returns:
        alist: shape=(num_level, B, ...)
    """
    new_order0 = len(alist[0])
    return [[alist[i][j] for i in range(len(alist))] for j in range(new_order0)]


# def cat_tensor_list(tensors, valids=None):
#     """
#         tensors: list(Tensor), the shape can be different, will use the max size as final shape
#     """
#     def get_max_shape(the_tensors):
#         t0 = tensors[0]
#         shapes = []
#         for t in tensors:
#             assert len(t.shape) == len(t0.shape), f"tensor's shape in give tensors must be match, got {t.shape} vs {t0.shape}"
#             shapes.append(t.shape)
#         shapes = torch.LongTensor(shapes)
#         return shapes.max(dim=0)[0].tolist()
#     max_shape = [len(tensors)] + get_max_shape(tensors)
#     new_tensor = torch.zeros(*max_shape).type_as(t0).to(t0.device)
#     torch.zeros(, dtype=torch.bool, device=t0.device)
#     for i, t, valid in enumerate(zip(tensors, valids)):
#         index = [i] + [slice(s) for s in t.shape]  # class slice(stop) or class slice(start, stop[, step])
#         index = tuple(index)   # must be tuple, tuple and list as index have diff meaning
#         # new_tensor[i, :t.shape[0], :t.shape[1], ....]
#         new_tensor[index] = t
#     return new_tensor


def sample_by_flag(feats, chosens_flag, num_max, ret_valid=True):
    """
    Args:
        feats: [k, (xxx, C1, C2, ...)], xxx can be H, W
        chosens_flag: (num_gt, xxx)
        num_max: int
    Return:
        feats: [k, (num_gt, num_max, C1, C2, ...)]
        valid: (num_gt, num_max)
    """
    num_gt, xxx = chosens_flag.shape[0], chosens_flag.shape[1:]

    ret_feats = []
    for feat in feats:
        xxx2, C = feat.shape[:len(xxx)], feat.shape[len(xxx):]
        assert xxx == xxx2
        ret_feat = torch.zeros(num_gt, num_max, *C).to(feat.device)
        ret_feats.append(ret_feat)

    if ret_valid:
        valid = torch.ones(num_gt, num_max).bool().to(feats[0].device)
    for i, chosen in enumerate(chosens_flag):  # (xxx)
        for feat, ret_feat in zip(feats, ret_feats):   # (xxx, *C), (num_gt, num_max, *C)
            C = ret_feat.shape[2:]
            chosen_feat_per_gt = feat[chosen].reshape(-1, *C)  # (num_max, C1, C2, ...)
            ret_feat[i, :len(chosen_feat_per_gt)] = chosen_feat_per_gt
        if ret_valid:
            num_chosen = chosen.long().sum()
            assert len(chosen_feat_per_gt) == num_chosen
            valid[i, num_chosen:] = False
    if ret_valid:
        return ret_feats, valid
    else:
        return ret_feats


def stack_feats(feats):
    """
    Args:
        feats: [B, (n, c)]
    Returns:
        feats: (B, N, c)
        valid: (B, N)
    """
    max_num = max([len(f) for f in feats])
    s0 = feats[0].shape
    for f in feats:
        assert f.shape[1:] == s0[1:], f"{f.shape} vs {s0}"

    shape = (max_num,) + feats[0].shape[1:]
    new_feats = []
    valids = []
    for feat in feats:
        new_feat = torch.zeros(shape, dtype=feat.dtype).to(feat.device)
        valid = torch.zeros(max_num, dtype=torch.bool).to(feat.device)
        new_feat[:len(feat)] = feat
        valid[: len(feat)] = 1
        new_feats.append(new_feat)
        valids.append(valid)
    return torch.stack(new_feats), torch.stack(valids)


def fill_list_to_tensor(alist, default_value=-1):
    max_l = max([len(l) for l in alist])
    data = torch.empty(len(alist), max_l, *alist[0].shape[1:]).type_as(alist[0]).to(alist[0])
    data[:] = default_value
    for i, l in enumerate(alist):
        data[i, :len(l)] = l
    return data


def group_by_label(data, labels):
    assert len(labels.shape) == 1
    labels = labels.cpu().numpy().tolist()
    label2data = defaultdict(list)
    for label, d in zip(labels, data):
        label2data[label].append(d)
    return {l: torch.stack(data) for l, data in label2data.items()}


def groups_by_label(datas, labels):
    assert len(labels.shape) == 1
    assert isinstance(datas, (tuple, list))
    labels = labels.cpu().numpy().tolist()
    multi_label2data = [defaultdict(list) for _ in datas]
    for i, label in enumerate(labels):
        for label2data, data in zip(multi_label2data, datas):
            label2data[label].append(data[i])
    return [{l: torch.stack(data) for l, data in label2data.items()}
            for label2data in multi_label2data]


def grid_sample(feat, chosen_pts, align_corners):
    """
    # (B=1, num_gt_pts, num_chosen, 2)
    Args:
        feat: shape=(B, C, H, W)
        chosen_pts:  shape=(B, num_gts, num_chosen, 2)
    Returns:
    """
    if align_corners:
        # [0, w-1] -> [-1, 1]
        grid_norm_func = lambda xy, wh: 2 * xy / (wh - 1) - 1
        padding_mode = 'zeros'
    else:
        # [-0.5, w-1+0.5] -> [-1, 1]
        # x -> x' => x' = (2x+1) / w - 1
        grid_norm_func = lambda xy, wh: (2 * xy + 1) / wh - 1  # align_corners=False
        padding_mode = 'border'
    h, w = feat.shape[2:]
    WH = feat.new_tensor([w, h])
    chosen_pts = grid_norm_func(chosen_pts, WH)
    return F.grid_sample(feat, chosen_pts, align_corners=align_corners, padding_mode=padding_mode)


class Statistic(object):
    def __init__(self):
        self.iter = {}

    def sum(self, name, x):
        """
        x: shape=(n, ...)
        """
        if not hasattr(self, name):
            setattr(self, f"{name}", [0, 0])
        value, count = getattr(self, f"{name}")
        value = value + x.sum(dim=0)
        count += len(x)
        setattr(self, name, [value, count])
        return value, count

    def mean(self, name, x):
        s, c = self.sum(name, x)
        return s / c

    def mean_scalar(self, name, x):
        if not hasattr(self, name):
            n = 1
            setattr(self, f"{name}", (x, n))
        else:
            old_x, n = getattr(self, f"{name}")
            x = old_x + x
            n += 1
            setattr(self, f"{name}", (x, n))
        return x / n

    def max_scalar(self, name, x):
        if not hasattr(self, name):
            setattr(self, f"{name}", x)
        else:
            old_x = getattr(self, f"{name}")
            x = max(old_x, x)
            setattr(self, f"{name}", x)
        return x

    def min_scalar(self, name, x):
        if not hasattr(self, name):
            setattr(self, f"{name}", x)
        else:
            old_x = getattr(self, f"{name}")
            x = min(old_x, x)
            setattr(self, f"{name}", x)
        return x

    def print_mean(self, name, x, log_interval=1):
        self.print(name, self.mean(name, x), log_interval,
                   prefix='[Statistic mean]')

    def log_mean_exp(self, name, x, eps=1e-7):
        x = (x+eps).log()
        x = self.mean(name, x)
        return x.exp() - eps

    def print_log_mean_exp(self, name, x, log_interval=1):
        self.print(name, self.log_mean_exp(name, x), log_interval,
                   prefix='[Statistic log_exp_mean count]')

    def count(self, name, data, momentum=0.99):
        """
            data: shape=(n)
        """
        import numpy as np
        data = self.to_numpy(data)
        count = np.bincount(data)
        if not hasattr(self, name):
            setattr(self, f"{name}", count)
        else:
            pre_count = getattr(self, f"{name}")
            if len(pre_count) < len(count):
                pre_count = np.array(pre_count.tolist() + [0] * (len(count) - len(pre_count)))
            if len(count) < len(pre_count):
                count = np.array(count.tolist() + [0] * (len(pre_count) - len(count)))
            count = count * (1 - momentum) + pre_count * momentum
            setattr(self, f"{name}", count)
        return count

    def print_count(self, name, data, momentum=0.99, log_interval=1):
        self.print(name, self.count(name, data, momentum), log_interval,
                   prefix='[Statistic count]')

    def ema(self, name, data, momentum=0.99):
        data = self.to_numpy(data)
        x = data.mean(axis=0)
        if not hasattr(self, name):
            setattr(self, f"{name}", x)
        else:
            pre_x = getattr(self, f"{name}")
            x = pre_x * momentum + x * (1 - momentum)
            setattr(self, f"{name}", x)
        return x

    def print_ema(self, name, data, momentum=0.99, log_interval=1):
        self.print(name, self.ema(name, data, momentum), log_interval,
                   prefix='[Statistic ema]')

    def c_ema(self, name, data, momentum=0.99):
        import numpy as np
        res = []
        for x in data:
            if len(x) > 0:
                res.append(np.mean(x))
            else:
                res.append(None)
        x = res

        if not hasattr(self, name):
            setattr(self, f"{name}", x)
        else:
            res = []
            pre_x = getattr(self, f"{name}")
            for i in range(len(x)):
                if pre_x[i] is not None and x[i] is not None:
                    res.append(pre_x[i] * momentum + x[i] * (1 - momentum))
                elif pre_x[i] is None:
                    res.append(x[i])
                elif x[i] is None:
                    res.append(pre_x[i])
                else:
                    raise ValueError(f"{x[i]} {pre_x[i]}")
            setattr(self, f"{name}", res)
        return res

    def print_c_ema(self, name, data, momentum=0.99, log_interval=1):
        self.print(name, self.c_ema(name, data, momentum), log_interval,
                   prefix='[Statistic c ema]')

    def sum_list(self, name, data):
        """data: [[], [...]]"""
        res = []
        for d in data:
            if len(d) == 0:
                res.append((0, 0))
            else:
                res.append((np.sum(d), len(d)))
        res = np.array(res)  # (n, 2), 2 is sum and count

        if not hasattr(self, name):
            setattr(self, f"{name}", res)
        else:
            last_res = getattr(self, f"{name}")
            res = last_res + res
            setattr(self, f"{name}", res)
        return res

    def mean_list(self, name, data, eps=1e-8):
        res = self.sum_list(name, data)
        return res[:, 0] / (res[:, 1] + eps)

    def print_mean_list(self, name, data, log_interval=1):
        self.print(name, self.mean_list(name, data), log_interval,
                   prefix='[Statistic mean list]')

    def log_mean_exp_list(self, name, data, eps=1e-7):
        data = [[np.log(x+eps) for x in d] for d in data]
        data = self.mean_list(name, data)
        return [np.exp(d) - eps for d in data]

    def print_log_mean_exp_list(self, name, data, log_interval=1):
        self.print(name, self.log_mean_exp_list(name, data), log_interval,
                   prefix='[Statistic log_exp_mean list]')

    def print(self, name, value, log_interval, prefix):
        _iter = self.iter.get(name, 0) + 1
        if (_iter - 1) % log_interval == 0:
            logger = get_root_logger()
            logger.info(f"{prefix} {name}: {value}")
        self.iter[name] = _iter

    def to_numpy(self, data):
        import numpy as np
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, (tuple, list)):
            return np.array(data)
        else:
            raise TypeError


class MultiList:
    """
    Example:
        > ml = MultiList()
        > ml.append(1, 2, 3)
        > ml.append(1, 2, 3)
        > ml.data
        [[1, 1], [2, 2], [3, 3]
    """

    def __init__(self):
        self.data = []

    def append(self, *args):
        if len(self.data) == 0:
            self.data = [[] for arg in args]
        assert len(args) == len(self.data)
        for l, arg in zip(self.data, args):
            l.append(arg)

    def apply(self, fn):
        return [fn(l) for l in self.data]


# utils type for passing argument

class MultiListData(object):
    class Data(object):
        pass

    def __init__(self, *args, **kwargs):
        self.other_data = None
        self.cfg = dict()

    def _get_all_attr(self):
        attrs = []
        for k, v in self.__dict__.items():
            if k.startswith('_') or k in ['other_data', 'cfg']:
                continue
            attrs.append((k, v))
        return attrs

    def __len__(self):
        attrs = self._get_all_attr()
        lens = [len(v) for k, v in attrs if v is not None]
        assert len(lens) > 0, "have not list data in multi list data"
        l = lens[-1]
        for i, ll in enumerate(lens[:-1]):
            assert l == ll, f'length in MultiListData must match but ' \
                            f'got {(attrs[i][0])}({ll}) vs {(attrs[-1][0])}({l})'
        return l

    def get(self, item, *args, **kwargs):
        return self.__getitem__(item)

    def __getitem__(self, item):
        data = MultiListData.Data()
        data.other_data = self.other_data
        data.cfg = self.cfg

        attrs = self._get_all_attr()
        for attr_name, attr_values in attrs:
            attr_value = None if attr_values is None else attr_values[item]
            setattr(data, attr_name, attr_value)
        return data

    def apply(self, fn):
        attr_names = self._get_all_attr()
        for attr_name, attr_values in attr_names:
            setattr(self, attr_name, fn(attr_values))


class Input(MultiListData):
    def __init__(self, img_metas, gt_r_pts=None, gt_pts=None, gt_labels=None,
                 gt_pts_ignore=None, gt_true_bboxes=None,
                 gt_weights=None, refined_geos=None):
        self.img_metas = img_metas            # [B, {}]
        self.gt_r_pts = gt_r_pts              # [B, (mum_gt, num_r, 2)]
        self.gt_pts = gt_pts                  # [B, (num_gt, 2)]
        self.gt_labels = gt_labels            # [B, (num_gt,)]
        self.gt_pts_ignore = gt_pts_ignore    # [B, (num_gt_ignore,)] or None
        self.gt_true_bboxes = gt_true_bboxes  # [B, (num_gt, 4)] or None

        self.gt_weights = gt_weights          # [B, (num_gt,)] or None
        self.refined_geos = refined_geos      # [B, (num_gt, x)] or None
        self.chosen_lvl = None                # [B, (num_gt,)] or None
        super(Input, self).__init__()

    def get(self, img_id, lvl=None):
        """
            return:
                data.gt_pts: shape=(num_gt_lvl, 2)
                ...
        """
        input_data_img = super(Input, self).get(img_id)
        if lvl is None:
            return input_data_img

        # split data of this lvl
        valid = input_data_img.chosen_lvl == lvl

        data = MultiListData.Data()
        data.other_data = input_data_img.other_data
        data.cfg = input_data_img.cfg
        data.img_metas = input_data_img.img_metas
        data.gt_pts_ignore = input_data_img.gt_pts_ignore

        attrs = self._get_all_attr()
        for attr_name, attr_values in attrs:
            if attr_name not in ["img_metas", "cfg", "other_data", 'gt_pts_ignore']:
                attr_value = None if attr_values is None else attr_values[img_id][valid]
                setattr(data, attr_name, attr_value)
        return data

    def set(self, img_id, lvl, data_list, out_data_list=None):
        """
            data_list: [k, num_lvl=1, (num_gt_lvl, ....)]
            out_data_list:  [k, num_lvl=1, (num_gt, ....)]
        """
        chosen_lvl_img = self.chosen_lvl[img_id]
        num_gt = len(chosen_lvl_img)

        data0 = data_list[0]
        for data in data_list[1:]:
            assert len(data0) == len(data), f"{len(data0)} vs {len(data)}"

        if out_data_list is None or len(out_data_list) == 0:
            out_data_list = [[torch.zeros(num_gt, *data.shape[1:]).type_as(data).to(data.device)]
                             for data in data_list]  # [k, num_lvl=1, (num_gt, ....)]

        lvl_valid = chosen_lvl_img == lvl
        for k, data in enumerate(data_list):
            out_data_list[k][0][lvl_valid] = data
        return out_data_list


class CascadeData(MultiListData):
    def __init__(self, refine_geos=None, refine_pts=None, chosen_pts=None):
        super(CascadeData, self).__init__()
        self.cfg["refine_geo_type"] = 'bbox'
        self.refined_geos = refine_geos
        self.refine_pts = refine_pts
        # self.refine_scores = refine_scores
        # self.not_refine = not_refine
        self.chosen_pts = chosen_pts


class PtAndFeat(object):
    def __init__(self, pts=None, valid=None, img_len=None):
        """"""
        self.pts = pts
        self.valid = valid
        self.img_len = img_len

        self.cls_feats = None
        self.ins_feats = None
        self.cls_outs = None
        self.ins_outs = None
        self.cls_prob = None
        self.ins_prob = None

    def split_each_img(self):
        """
        Returns:
            res: list[PtAndFeat], pts and feats info of each image
        """
        def split_data_each_img(datas, img_lens):
            """
            Args:
                datas: [lvl, (num_all_img, ...)]
                img_lens:
            Returns:
                res: [B, lvl, (num_per_img, ...)]
            """
            res = [[] for lvl in range(len(img_lens))]
            for lvl, img_len in enumerate(img_lens):
                i = 0
                for l in img_len:
                    res[lvl].append(datas[lvl][i:i + l])
                    i += l
            B = len(res[0])
            for r in res:
                assert len(r) == B
            return [[res[lvl][b] for lvl in range(len(res))] for b in range(B)]

        assert self.pts is not None and self.img_len is not None
        res = None
        for key, value in self.__dict__.items():
            if key in ["img_len"]:
                continue
            if value is None:
                continue
            value_list = split_data_each_img(value, self.img_len)
            if res is None:
                res = [PtAndFeat() for _ in value_list]
            for i, v in enumerate(value_list):
                res[i].__setattr__(key, v)
        return res


if __name__ == '__main__':
    d = CascadeData()
    print(d._get_all_attr())
    # print(len(d))
    print(d.__dict__.items())

    a = torch.rand(1, 2, 3, 4)
    b = torch.rand(4, 3, 2, 1)
    c = cat_tensor_list((a, b))
    print(a, b)
    print(c)
