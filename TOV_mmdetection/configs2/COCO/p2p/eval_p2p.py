from huicv.exp.mmdet_exp import *


class Exp(BaseExp):
    def train_name(self, cfg):
        batch = cfg['data']['samples_per_gpu']
        lr = cfg['optimizer']['lr']
        return f"adam{lr}_1x_{batch}b"

    def get_cmds(self, cfg, g_cfg):
        default_g_cfg = dict(
            port=10000,
            gpus=list(range(8)),
            resume=False,

            dataset='COCO/p2p',
            noise='',
            cfg_name='p2p_r50_fpn_1x_fl_sl1_coco400',
        )
        g_cfg = update_dict(default_g_cfg, g_cfg)

        # general setting
        port, gpus = g_cfg['port'], g_cfg['gpus']

        cfg_path = f"configs2/{g_cfg['dataset']}/{g_cfg['cfg_name']}.py"
        save_dir = f"../TOV_mmdetection_cache/work_dir/{g_cfg['dataset']}/"
        exp_setting = f"{self.train_name(cfg)}{len(gpus)}g"
        work_dir = f"{save_dir}/{g_cfg['noise']}/{g_cfg['cfg_name']}/{exp_setting}/"

        resume = f"--resume-from {work_dir}/latest.pth " if g_cfg['resume'] else ""
        gpu_str = ",".join([str(s) for s in gpus])

        cmd = f'CUDA_VISIBLE_DEVICES={gpu_str} PORT={port} tools/dist_train.sh {cfg_path} {len(gpus)}' \
              f' --work-dir {work_dir} {resume} --cfg-options {cfg_to_str(cfg)}'
        return [cmd]


def general_cfg():
    return dict(
        data=dict(samples_per_gpu=8),
        optimizer=dict(lr=1e-4),
        evaluation=dict(do_final_eval=True)
    )


def inference_cfg(pseudo_wh, nms_iou):
    cfg = dict(model=dict(test_cfg=dict(
        pseudo_wh=f"'({pseudo_wh},{pseudo_wh})'",
        nms=dict(iou_threshold=nms_iou),
    )))
    return cfg


args = Exp.parse_args()
for cfg_name in [
    'p2p_r50_fpn_1x_fl_sl1_coco400_coarse',
    # 'p2p_r50_fpn_1x_fl_sl1_coco400',
    # 'p2p_r50_fpns16_1x_fl_sl1_coco400',
    # 'p2p_r50_fpns4_1x_fl_sl1_coco',  # 32
    # 'p2p_r50_fpn_1x_fl_sl1_coco'     # 32
]:
    g_cfg = dict(resume=True, cfg_name=cfg_name)
    for s, iou in [(32, 0.05), (32, 0.0)]:
        cfg = general_cfg()
        cfg = update_dict(cfg, inference_cfg(pseudo_wh=s, nms_iou=iou))
        Exp().run(cfg, g_cfg, args)
