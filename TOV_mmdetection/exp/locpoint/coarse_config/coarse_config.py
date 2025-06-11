import os
import json

"""
1. 我所需要其实是一个查询系统，根据输入属性设置查询相关的实验在哪里。
2. 创建新的实验时候，判断相同配置的实验是否跑了，如果跑过了，提示是否新建目录重跑或者覆盖原有目录重跑，或者不跑，返回查询结果。

3. 目录可以把关键的实验属性写进去，但是过于细节的就无法写进去，否则目录太长没法看，其他属性通过上面说的查询系统建立
目录属性，写一个映射，根据属性返回目录名字
   get_exp_setting_dir(cfg) => str
4. 如果目录属性相同，而其他的属性不同，那么则在目录属性后面添加_i{number} number是不同的设置的实验的个数，从1开始
5. 如果属性完全相同，而要创建新的目录，那么则在目录后面添加_{number}, number是当前的个数，从1开始

6. 属性记录文件，和mmdetection会复制一个文件一样，这个文件记录
自定义修改涉及的到的值
新的自定义的属性，包括实验组之间的属性记录


CoarsePoint的几个主要的任务:
1. 把命令形成脚本跑出来 => cmd
2. 把关键变量提取出来 => attribution
3. 中间变量提成property => property
4. Var引入，结合ctx.cfgs 连接组间关系,依赖属性 =>
"""


def join_list(sep, alist):
    return sep.join([str(_) for _ in alist])


class Var(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, cfg):
        return self.func(cfg)


class LastAttr(Var):
    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, cfg):
        i = cfg.cfg_idx
        cfgs = cfg.ctx.cfgs
        return getattr(cfgs[i - 1], self.attr_name)


class ExpConfig(object):
    def __init__(self, cfg_idx, ctx):
        self.cfg_idx = cfg_idx
        self.ctx = ctx

    def update(self, kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), k
            setattr(self, k, v)

    def __getattribute__(self, item):
        x = super(ExpConfig, self).__getattribute__(item)
        if isinstance(x, Var):
            return x(self)
        return x

    def save_as_json(self, filepath):
        adict = {}
        for k, v in self.__dict__.items():
            if k != 'ctx':
                adict[k] = self.__getattribute__(k)
        json.dump(adict, open(filepath, 'w'), separators=(',', ':'))

    @property
    def config_dir(self):
        return os.path.split(self.config_file)[0]

    @property
    def config_name(self):
        return os.path.split(self.config_file)[1][:-3]  # remove .py

    @property
    def pre_solve(self):
        return self.presolve[0].format(*self.presolve[1:])

    @property
    def ann_info(self):
        return "{}/{}/{}".format(self.noise_type, self.corner, self.pre_solve)

    @property
    def exp_setting_dir(self):
        lr = self.cfg_options['optimizer.lr']
        return f"lr{lr}_1x_{len(self.gpus)}g"

    @property
    def work_dir(self):
        return f"{self.save_root}/{self.ann_info}/{self.config_name}/{self.exp_setting_dir}/{self.round_dir}/"

    @property
    def output_ann(self):
        return f"{self.work_dir}/coarse_refine.json"

    @property
    def checkpoint(self):
        return f"{self.work_dir}/latest.pth"


class TinyPersonConfig(ExpConfig):
    def __init__(self, cfg_idx, ctx):
        # common config
        self.gpus = [0, 1]
        self.port = 10000
        self.ann_root = "data/tiny_set/mini_annotations/coarse_gen/"
        self.save_root = "../TOV_mmdetection_cache/work_dir/TinyPerson/coarsepoint/"

        self.noise_type = "noise_uniform_1"
        self.corner = "corner_sw640_sh512_old"
        self.presolve = ("pseuw{}h{}", 16, 16)
        self.config_file = "faster_rcnn_r50_fpn_1x_TinyPerson640_coarsept.py"
        self.cfg_options = {
            "optimizer.lr": 0.01,
        }

        # round config
        self.round_dir = f"round{cfg_idx}"
        self.input_ann = f"{self.ann_root}/{self.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json"
        # round config: for inference, to generate box for refinement
        self.true_ann = "data/tiny_set/mini_annotations/tiny_set_train_sw640_sh512_all_erase.json"
        self.init_coarse_ann = f"{self.ann_root}/{self.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json"
        self.train_img_root = "data/tiny_set/erase_with_uncertain_dataset/train/"
        self.max_grow_rate = 1.75
        super(TinyPersonConfig, self).__init__(cfg_idx, ctx)


class TinyCOCOConfig(ExpConfig):
    def __init__(self, cfg_idx, ctx):
        self.gpus = [0, 1, 2, 3]
        self.port = 10000
        self.ann_root = "data/coco/resize/coarse_gen_annotations/"
        self.save_root = "../TOV_mmdetection_cache/work_dir/TinyCOCO/coarsepoint/"

        self.noise_type = 'noise_uniform_1'
        self.corner = ""
        self.presolve = ("pseuw{}h{}", 16, 16)
        self.config_file = "faster_rcnn_r50_fpn_1x_TinyPerson640_coarsept.py"
        self.cfg_options = {
            "optimizer.lr": 0.04,
            "data.samples_per_gpu": 16,
        }

        # round config
        self.round_dir = f"round{cfg_idx}"
        self.input_ann = f"{self.ann_root}/{self.ann_info}/instances_train2017_100x167_coarse.json"
        # round config: for inference, to generate box for refinement
        self.true_ann = "data/coco/resize/annotations/instances_train2017_100x167.json"
        self.init_coarse_ann = f"{self.ann_root}/{self.ann_info}/instances_train2017_100x167_coarse.json"
        self.train_img_root = "data/coco/resize/images_100x167_q100/"
        self.max_grow_rate = 1.75

        super(TinyCOCOConfig, self).__init__(cfg_idx, ctx)

    @property
    def exp_setting_dir(self):
        lr = self.cfg_options['optimizer.lr']
        batch_size = self.cfg_options["data.samples_per_gpu"]
        return f"lr{lr}_1x_{batch_size}b{len(self.gpus)}g"


class VisDronePersonConfig(ExpConfig):
    def __init__(self, cfg_idx, ctx):
        self.gpus = [0, 1, 2, 3]
        self.port = 10000
        self.ann_root = "data/visDrone/coarse_gen_annotations/"
        self.save_root = "../TOV_mmdetection_cache/work_dir/visDronePerson/coarsepoint/"

        # self.noise_type = 'noise_uniform_1'
        self.noise_type = 'noise_rg0.0_0.25_1'
        self.corner = "corner_sw640_sh640"
        self.presolve = ("pseuw{}h{}", 32, 32)
        self.config_file = ""
        self.cfg_options = {
            "optimizer.lr": 0.005,
            "data.samples_per_gpu": 2,
        }

        # round config
        self.round_dir = f"round{cfg_idx}"
        self.input_ann = f"{self.ann_root}/{self.ann_info}/VisDrone2018-DET-train-person-w640h640ow100oh100_coarse.json"
        # round config: for inference, to generate box for refinement
        self.true_ann = "data/visDrone/coco_fmt_annotations/corner/VisDrone2018-DET-train-person-w640h640ow100oh100.json"
        self.init_coarse_ann = f"{self.ann_root}/{self.ann_info}/VisDrone2018-DET-train-person-w640h640ow100oh100_coarse.json"
        self.train_img_root = "data/visDrone/VisDrone2018-DET-train/images/"
        self.max_grow_rate = 1.35

        super(VisDronePersonConfig, self).__init__(cfg_idx, ctx)

    @property
    def exp_setting_dir(self):
        lr = self.cfg_options['optimizer.lr']
        batch_size = self.cfg_options["data.samples_per_gpu"]
        return f"lr{lr}_1x_{batch_size}b{len(self.gpus)}g"


from copy import deepcopy


def get_exps_config(common_config: dict, exps_config: list):
    final_exps_config = []
    for cfg in exps_config:
        final_cfg = deepcopy(common_config)
        final_cfg.update(cfg)
        final_exps_config.append(final_cfg)
    return final_exps_config


def build_configs(ctx, ConfigClass, exps_config):
    cfgs = []
    for i, exp_config in enumerate(exps_config):
        cfg = ConfigClass(i, ctx)
        cfg.update(exp_config)
        cfgs.append(cfg)
    return cfgs
