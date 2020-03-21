# ############## add by gongyuqi ###################################################3

from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import torch
import math
from functools import partial


class BalanceNormalRandomSampler(Sampler):
    r"""
    changed from Random Sampler, 
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __is_normal(self, image_info):
        """
        generated image and pure background image are all not_normal image
        :param image_info:
        :return:
        """
        if 'is_generate' in image_info:
            if image_info['is_generate']:  # generated image
                return False
        ann_ids = self.data_source.coco.getAnnIds(imgIds=image_info['id'])
        if len(ann_ids) == 0:              # pure background image
            return False
        return True

    def __init__(self, data_source, replacement=False, num_samples=None, normal_ratio=0.5):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.foreground_ratio = normal_ratio

        self.normal = []
        self.not_normal = []
        self.dict_imgid_to_sampleid = {}

        print("dataset have image count is", len(data_source))

        # for id in range(len(data_source.ids)):
        #     self.dict_imgid_to_sampleid[data_source.ids[id]] = id   # ids map sample_id to image_id
        for sample_idx in range(len(data_source)):
            image_info = data_source.get_img_info(sample_idx)
            if self.__is_normal(image_info):
                self.normal.append(sample_idx)
            else:
                self.not_normal.append(sample_idx)

        self.not_normal = torch.LongTensor(self.not_normal)
        self.normal = torch.LongTensor(self.normal)

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        assert 0 < normal_ratio <= 1, 'normal ratio range must in (0, 1].'

        if self.num_samples is None:
            # self.num_samples = len(self.data_source)
            self.num_samples = int(len(self.normal) / normal_ratio)

        assert self.num_samples - len(self.normal) <= len(self.not_normal), \
            'no enough not-normal image in dataset, need {}, have {}.'.\
                format(self.num_samples - len(self.normal), len(self.not_normal))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self, generator=None):
        """
        -> sample_idx
        :return:
        """
        randint = partial(torch.randint, generator=generator) if generator is not None else torch.randint
        randperm = partial(torch.randperm, generator=generator) if generator is not None else torch.randperm

        num_normal = len(self.normal)
        num_not_normal = self.num_samples - len(self.normal)
        if self.replacement:
            chosen_normal = self.normal[randint(high=len(self.normal), size=(num_normal,), dtype=torch.int64)]
            chosen_not_normal = self.not_normal[randint(high=len(self.not_normal), size=(num_not_normal,), dtype=torch.int64)]
            chosen = torch.cat([chosen_normal, chosen_not_normal], 0)
        else:
            not_normal_idx = randperm(len(self.not_normal))[:num_not_normal]
            chosen_not_normal = self.not_normal[not_normal_idx]
            chosen = torch.cat([self.normal, chosen_not_normal], 0)
        chosen = chosen[randperm(len(chosen))]
        return iter(chosen.tolist())

    def __len__(self):
        return self.num_samples


class SamplerToDistributedSampler(Sampler):
    """
    changed from DistributedSampler
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        sample.__iter__ must support set random generator by input argument
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.sampler) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(self.sampler.__iter__(g))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
