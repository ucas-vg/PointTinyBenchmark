from exp.locpoint.coarse_point_manager import *


class SemiTinyPersonConfig(TinyPersonConfig):
    def __init__(self, cfg_idx, ctx):
        super(SemiTinyPersonConfig, self).__init__(cfg_idx, ctx)
        # common config
        self.gpus = [0, 1]
        self.ann_root = "data/tiny_set/mini_annotations/coarse_gen/"
        self.save_root = "../TOV_mmdetection_cache/work_dir/TinyPerson/semi-coarsepoint/"

        self.noise_type = "noise_uniform_1"
        self.corner = "corner_sw640_sh512_old"
        self.presolve = ("pseuw{}h{}", 16, 16)
        self.config_file = "xxx.py"
        self.cfg_options = {
            "optimizer.lr": 0.01,
        }
        self.fully_ratio = 0.8

        # round config
        self.round_dir = f"round{cfg_idx}"
        self.input_ann = f"{self.ann_root}/{self.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json"
        # round config: for inference, to generate box for refinement
        self.true_ann = "data/tiny_set/mini_annotations/tiny_set_train_sw640_sh512_all_erase.json"
        self.init_coarse_ann = f"{self.ann_root}/{self.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json"
        self.train_img_root = "data/tiny_set/erase_with_uncertain_dataset/train/"
        self.max_grow_rate = 1.75

    @property
    def exp_setting_dir(self):
        lr = self.cfg_options['optimizer.lr']
        return f"semi{self.fully_ratio}_lr{lr}_1x_{len(self.gpus)}g"

    @property
    def output_ann(self):
        return f"{self.work_dir}/coarse_semi{self.fully_ratio}_refine.json"


class SemiCorasePointExp(CorasePointExp):
    def __init__(self, ConfigClass):
        net = {
            'RepPoint': "configs2/TinyPerson/coarsepoint/reppoints_moment_r50_fpns4_gn-neck+head_1x_TinyPerson640_coarsept.py"
        }
        exps_config = [
            {
                "round_dir": "round0",
                "input_ann": Var(
                    lambda cfg: f"{cfg.ann_root}/{cfg.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse_semi{cfg.fully_ratio}.json"),
                "config_file": net['RepPoint'],
            },
            {
                "round_dir": "round1_Rp",
                "input_ann": LastAttr("output_ann"),  # equal to last output_ann
                "config_file": net['RepPoint'],
            },
            {
                "round_dir": "round2_Rp-Rp",
                "input_ann": LastAttr("output_ann"),
                "config_file": net['RepPoint'],
            },
        ]
        super(SemiCorasePointExp, self).__init__(ConfigClass, exps_config)

        cfg = self.cfgs[0]
        self.steps.append(
            "PYTHONPATH=.:$PYTHONPATH python huicv/coarse_utils/generate_semi_annotation.py \\\n"
            f"\t {cfg.ann_root}/{cfg.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse.json\\\n"
            f"\t {cfg.ann_root}/{cfg.ann_info}/tiny_set_train_sw640_sh512_all_erase_coarse_semi{cfg.fully_ratio}.json\\\n"
            f"\t --fully_ratio {cfg.fully_ratio}"
        )


if __name__ == "__main__":
    exp_group = SemiCorasePointExp(SemiTinyPersonConfig)
    exp_group.build_cmd()
    exp_group.print_cmd()
    # exp_group.train()
