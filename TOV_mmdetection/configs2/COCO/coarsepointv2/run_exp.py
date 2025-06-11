import os
import argparse


class EXP:
    def set(self, k, cfg, ck, defalut=None, prefix='', suffix=''):
        x = ''
        if ck in cfg:
            x = f"{k}={cfg[ck]}"
        return f"{prefix}{x}{suffix}"

    def refine(self, config):
        cfg = config['refine']
        refine_base = {
            "refine0": "",
            "refine1": "model.bbox_head.point_refiner.merge_th=0.05 model.bbox_head.point_refiner.refine_th=0.05",
            "refine2": "model.bbox_head.point_refiner.merge_th=0.05 model.bbox_head.point_refiner.refine_th=0.05 "
                       "model.bbox_head.point_refiner.classify_filter=True",
            "refine2_2": "model.bbox_head.point_refiner.merge_th=0.1 model.bbox_head.point_refiner.refine_th=0.1 "
                         "model.bbox_head.point_refiner.classify_filter=True",
            "refine2_3": "model.bbox_head.point_refiner.merge_th=0.1 model.bbox_head.point_refiner.refine_th=0.2 "
                         "model.bbox_head.point_refiner.classify_filter=True",
            "refine2_4": "model.bbox_head.point_refiner.merge_th=0.2 model.bbox_head.point_refiner.refine_th=0.2 "
                         "model.bbox_head.point_refiner.classify_filter=True",
        }
        if 'sample' in cfg:
            r = cfg['sample']
            refine_sample = f'{self.set("model.bbox_head.refine_pts_extractor.pos_generator.radius", r, "pos_r")} ' \
                            f'{self.set("model.bbox_head.refine_pts_extractor.neg_generator.radius", r, "neg_r")} '
        else:
            refine_sample = ""
        return f"{refine_sample} {refine_base[cfg['type']]}"

    def refine_name(self, config):
        cfg = config['refine']
        if 'sample' in cfg:
            r = cfg['sample']
            sample = f"_r{r['pos_r']}_{r['neg_r']}"
        else:
            sample = ""
        return f"_{cfg['type']}{sample}"

    def train(self, config):
        train = config["train"]
        if 'sample' in train:
            t = train['sample']
            sample = f'{self.set("model.bbox_head.train_pts_extractor.pos_generator.radius", t, "pos_r")} ' \
                     f'{self.set("model.bbox_head.train_pts_extractor.neg_generator.radius", t, "neg_r")} ' \
                     f'{self.set("model.bbox_head.train_pts_extractor.neg_generator.class_wise", t, "neg_class_wise")} '
        else:
            sample = ""
        if 'loss_cfg' in train:
            t = train['loss_cfg']
            loss = f'{self.set("model.bbox_head.loss_cfg.with_mil_loss", t, "with_mil_loss")} ' \
                   f'{self.set("model.bbox_head.loss_mil.type", t, "mil_loss_type")} ' \
                   f'{self.set("model.bbox_head.loss_cfg.with_gt_loss", t, "with_gt_loss")} ' \
                   f'{self.set("model.bbox_head.loss_cfg.with_neg", t, "with_neg_loss")} ' \
                   f'{self.set("model.bbox_head.loss_mil.binary_ins", t, "binary_ins")} ' \
                   f'{self.set("model.bbox_head.loss_cfg.random_remove_rate", t, "drop")} '
        else:
            loss = ""

        if 'normal_cfg' in train:
            t = train['normal_cfg']
            normal = f'{self.set("model.bbox_head.normal_cfg.prob_cls_type", t, "prob_cls_type")} '
        else:
            normal = ""

        cascade = ""
        if 'cascade' in train:
            t = train['cascade']
            num_stages = t.get('num_stages', -1)
            if num_stages > 0:
                cascade += f'{self.set(f"model.bbox_head.num_stages", t, "num_stages")} '
                cascade += f"model.bbox_head.cpr_cfg_list='' "
            for k in ["refine_bag_policy", "gt_loss_type"]:
                cascade += f'{self.set(f"model.bbox_head.loss_cfg.{k}", t, k)} '
            for k in t:
                if k in ["refine_bag_policy", "gt_loss_type", "num_stages"]: continue
                cascade += f'{self.set(f"model.bbox_head.cascade_cfg.{k}", t, k)} '
        return f"{sample}{loss}{normal}{cascade}"

    def train_name(self, config):
        train = config["train"]
        if 'sample' in train:
            t = train['sample']
            sample = f"_r{t['pos_r']}_{t['neg_r']}"
            if 'neg_class_wise' in t and not t['neg_class_wise']:
                sample += '_bgneg'
        else:
            sample = "_r5_3"

        loss = "loss"
        if 'loss_cfg' in train:
            t = train['loss_cfg']
            loss += '0'
            if 'with_gt_loss' in t and t['with_gt_loss']:
                loss += 'gt'
            if 'with_mil_loss' in t and not t['with_mil_loss']:
                loss += '_nomil'
            if 'mil_loss_type' in t:
                m = {"MILLoss": "", "AllPosLoss": '_allpos'}
                loss += m[t["mil_loss_type"]]
            if 'with_neg_loss' in t and not t['with_neg_loss']:
                loss += "_noneg"
            if 'binary_ins' in t and t['binary_ins']:
                loss += '_2ins'
            if 'drop' in t:
                loss += f'_drop{t["drop"]}'
        else:
            loss += '0'

        normal = ''
        if 'normal_cfg' in train:
            t = train['normal_cfg']
            if 'prob_cls_type' in t:
                m = {'sigmoid': '', 'normed_sigmoid': '_NormSig'}
                normal += m[t['prob_cls_type']]

        cascade = ""
        if 'cascade' in train:
            t = train['cascade']
            if 'increase_r' in t and t['increase_r']:
                cascade += 'incR'
            if 'increase_r_step' in t and t['increase_r_step'] > 1:
                cascade += f'{t["increase_r_step"]}step'
            if 'num_stages' in t and t['num_stages'] > 0:
                cascade += f"{t['num_stages']}stage_"
            if 'gt_src' in t:
                m = {'refine': 'c1c_', 'gt_refine': 'c2c_', 'gt': 'c0c_'}
                cascade += m[t['gt_src']]
        return f'{cascade}{loss}{sample}{normal}'

    def default(self, cfg):
        if 'train' not in cfg:
            cfg['train'] = {}
        if 'refine' not in cfg:
            cfg['refine'] = {'type': 'refine0'}
        return cfg

    def run_pseudo_bbox(self, GPUS, NOISE, CORNER, SETTING, SAVE_DIR, CFG_DIR, PORT, WORK_DIR, refine_name):
        # experiment group setting
        # PORT=10001
        # GPUS=[0, 1, 2, 3]
        # LR, BATCH = 0.02, 8
        # CONFIG = "reppoints_moment_r50_fpns4_gn-neck+head_1x_tinycoco_coarsept"
        # gpu_str = ",".join([str(s) for s in GPUS])
        # prefix = f""
        # ANN_TYPE = f"{NOISE}/{CORNER}/{SETTING}"
        # SWORK_DIR = f"{SAVE_DIR}/coarsepoint/{ANN_TYPE}/{CONFIG}/lr{LR}_1x_{BATCH}b{len(GPUS)}g/"
        # cmd2 = f"export GPU={len(GPUS)} && export LR={LR} && export BATCH={BATCH} && export R=0 &&" \
        #        f" CUDA_VISIBLE_DEVICES={gpu_str} PORT={PORT} tools/dist_train.sh" \
        #        f" {CFG_DIR}/coarsepoint/{CONFIG}.py {len(GPUS)} --work-dir {SWORK_DIR}" \
        #        f" --cfg-options optimizer.lr={LR} data.samples_per_gpu={BATCH}" \
        #        f" data.train.ann_file={WORK_DIR}/instances_train2017_100x167{refine_name}.json"
        # return cmd2
        raise NotImplementedError

    def run_p2p_net(self, GPUS, NOISE, CORNER, SETTING, SAVE_DIR, CFG_DIR, PORT, WORK_DIR, refine_name):
        LR, BATCH = 1e-4, 8
        CONFIG = 'p2p_r50_fpn_1x_fl_sl1_coco400_coarse'  # change here
        gpu_str = ",".join([str(s) for s in GPUS])
        ANN_TYPE = f"{NOISE}/{CORNER}/{SETTING}"
        SWORK_DIR = f"{SAVE_DIR}/p2p_coarse/{ANN_TYPE}/{CONFIG}/adam{LR}_1x_{BATCH}b{len(GPUS)}g/"
        cmd2 = f"export GPU={len(GPUS)} && export LR={LR} && export BATCH={BATCH} &&" \
               f" CUDA_VISIBLE_DEVICES={gpu_str} PORT={PORT} tools/dist_train.sh" \
               f" {CFG_DIR}/p2p/{CONFIG}.py {len(GPUS)} --work-dir {SWORK_DIR}" \
               f" --cfg-options optimizer.lr={LR} data.samples_per_gpu={BATCH}" \
               f" data.train.ann_file={WORK_DIR}/instances_train2017{refine_name}.json" \
               f" model.test_cfg.nms.iou_threshold=0.01"
        return cmd2

    def run(self, cfg=dict(),
            resume=True, use_p2p=True, args=dict(), other_cpr_cfg="", noise='noise_rg-0-0-0.25-0.25_1',
            CFG_FILE="coarse_point_refine_r50_fpns4_1x_coco400"):
        cfg = self.default(cfg)
        NOISE = noise
        print(noise)
        CORNER = ""
        SAVE_DIR = "../TOV_mmdetection_cache/work_dir/COCO/"
        CFG_DIR = "configs2/COCO/"

        if NOISE == 'center':
            ORI_ANN = "data/coco/annotations/instances_train2017.json"
        elif NOISE == 'UFO2':
            ORI_ANN = "data/coco/pts_annotation_published/instances_train_val_2017_point.json"
        else:
            ORI_ANN = f'data/coco/coarse_gen_annotations/{NOISE}/pseuw16h16/instances_train2017_coarse.json'

        PORT = 10001
        GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
        # PORT = 9999
        # GPUS = [4, 5, 6, 7]

        LR, BATCH = 0.01, 8
        SETTING = f"{self.train_name(cfg)}_lr{LR}_1x_{BATCH}b{len(GPUS)}g"
        WORK_DIR = f"{SAVE_DIR}/coarsepointv2/{NOISE}/{CFG_FILE}/{SETTING}/"
        resume = f"--resume-from {WORK_DIR}/latest.pth " if resume else ""
        gpu_str = ",".join([str(s) for s in GPUS])
        prefix = f"GPU={len(GPUS)} && CUDA_VISIBLE_DEVICES={gpu_str} PORT={PORT}"
        refine_name = self.refine_name(cfg)
        cmd0 = f"{prefix} tools/dist_train.sh " \
               f"{CFG_DIR}/coarsepointv2/{CFG_FILE}.py {len(GPUS)} " \
               f"--work-dir {WORK_DIR} {resume}" \
               f"--cfg-options optimizer.lr={LR} data.samples_per_gpu={BATCH} {self.refine(cfg)} {self.train(cfg)} {other_cpr_cfg}" \
               f" data.train.ann_file={ORI_ANN}" \
               f" evaluation.save_result_file={WORK_DIR}/latest_result{refine_name}.json"

        cmd1 = f"python exp/tools/result2ann.py --ori_ann {ORI_ANN}" \
               f" --det_file {WORK_DIR}/latest_result{refine_name}.json" \
               f" --save_ann {WORK_DIR}/instances_train2017{refine_name}.json"

        cmd2_args = GPUS, NOISE, CORNER, SETTING, SAVE_DIR, CFG_DIR, PORT, WORK_DIR, refine_name
        if use_p2p:
            cmd2 = self.run_p2p_net(*cmd2_args)
        else:
            cmd2 = self.run_pseudo_bbox(*cmd2_args)
        cmd3 = f'python exp/tools/killgpu.py {gpu_str}'

        for i, cmd in enumerate([cmd0, cmd1, cmd2, cmd3]):
            if i >= args.start:
                if args.end < 0 or i < args.end:
                    print(f"# [cmd {i}]")
                    print(cmd)
                    print()
                    if args.run:
                        os.system(cmd)


parser = argparse.ArgumentParser()
parser.add_argument("--run", help="", action="store_true")
parser.add_argument("--start", help="", default=0, type=int)
parser.add_argument("--end", help="", default=-1, type=int)
args = parser.parse_args()

use_p2p = True

# r = 8
# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True, mil_loss_type='AllPosLoss')),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')

# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=False)),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')
#
# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True, with_neg_loss=False)),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')
#
# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True, with_mil_loss=False)),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')
#
# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True)),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')


# # may need modify LR and Batch
# for r in [25]:  # [6, 7, 8, 9]:
#     cfg = dict(
#         train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True)),
#         refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
#     )
#     EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')

# for r in [40]:
#     cfg = dict(
#         train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True)),
#         refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
#     )
#     EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpns4_1x_coco400')

# for num_stages, r_step in [(3, 3), (3, 4)]:  # [(3, 1), (3, 2), (2, 1)]:
#     r = 6
#     cfg = dict(
#         train=dict(sample=dict(pos_r=r, neg_r=r),
#                    cascade=dict(gt_src='gt_refine', num_stages=num_stages,
#                                 increase_r=True, increase_r_step=r_step)),
#         refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r)),
#     )
#     EXP().run(cfg, False, use_p2p, args,  CFG_FILE='cascade_coarse_point_refine_r50_fpn_1x_coco400')

# binary_ins / normed_sigmoid / drop / not neg_class_wise
r = 5
cfg = dict(
    train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True, binary_ins=True)),
    refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
)
EXP().run(cfg, False, use_p2p, args, noise='UFO2', CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')

# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True),
#                normal_cfg=dict(prob_cls_type='normed_sigmoid')),  # more sharp
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')

# for drop in [0.0, 0.6]:
#     cfg = dict(
#         train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True, drop=drop)),
#         refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r))
#     )
#     EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')
#
# cfg = dict(
#     train=dict(sample=dict(pos_r=r, neg_r=r), loss_cfg=dict(with_gt_loss=True)),
#     refine=dict(type='refine2_2', sample=dict(pos_r=r, neg_r=r, neg_class_wise=False))
# )
# EXP().run(cfg, False, use_p2p, args, CFG_FILE='coarse_point_refine_r50_fpn_1x_coco400')
