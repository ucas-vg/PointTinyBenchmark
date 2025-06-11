from exp.locpoint.coarse_config.coarse_config import *
from copy import deepcopy


class CorasePointExp(object):
    def __init__(self, ConfigClass, exps_config):
        self.cfgs = build_configs(self, ConfigClass, exps_config)
        self.steps = []

        self.stop_if_err = True
        self.nan_restart_time = 3

    def build_cmd(self):
        for i, cfg in enumerate(self.cfgs):
            gpu_config = f"GPU={len(cfg.gpus)} && CUDA_VISIBLE_DEVICES={join_list(',', cfg.gpus)}"
            base_train = f"{gpu_config} PORT={cfg.port} tools/dist_train.sh {cfg.config_file} {len(cfg.gpus)}"
            base_test = f"{gpu_config} PORT={cfg.port} tools/dist_test.sh {cfg.config_file} {cfg.checkpoint} {len(cfg.gpus)}"

            work_dir = cfg.work_dir
            max_grow_rate = cfg.max_grow_rate

            self.steps.extend([
                # step 1 train with train dataset, input: train_ann, work_dir, general config
                f'{base_train} \\\n\t--work-dir {work_dir} \\\n\t'
                f'--cfg-options {" ".join([f"{k}={v}" for k, v in cfg.cfg_options.items()])} \\\n\t'
                f'data.train.ann_file="{cfg.input_ann}"',

                # step 2 inference on train dataset and get result,
                # input: work_dir(same as 1), resume weight(generate by 1), file and image_root
                # of val set as train, general config
                # f'{base_train} \\\n\t--work-dir {work_dir} \\\n\t'
                # f'--cfg-options {" ".join([f"{k}={v}" for k, v in cfg.cfg_options.items()])} \\\n\t'
                # f'evaluation.do_final_eval=True \\\n\t'
                # f'data.train.ann_file="{cfg.input_ann}" \\\n\t'
                # f'data.val.ann_file="{cfg.true_ann}" \\\n\t'
                # f'data.val.img_prefix="{cfg.train_img_root}" \\\n\t'
                # f'--resume-from {cfg.checkpoint}.pth',

                # f'mv exp/latest_result.json {work_dir}/latest_results.bbox.json',

                f'{base_test} \\\n\t--work-dir {cfg.save_root}/tmp \\\n\t'
                f'--cfg-options {" ".join([f"{k}={v}" for k, v in cfg.cfg_options.items()])} \\\n\t'
                f'data.test.img_prefix="{cfg.train_img_root}" \\\n\t'
                f'data.test.ann_file="{cfg.true_ann}" \\\n\t'
                f'--format-only --eval-options jsonfile_prefix="{work_dir}/latest_results"',

                # step 3 use the result generate new json file, input: origin_ann, latest_result(generate by 2),
                # image_dir, save_path(same dir as latest result)
                f'PYTHONPATH=.:$PYTHONPATH python huicv/coarse_utils/generate_new_bbox_json_file_corner.py \\\n\t'
                f'{cfg.init_coarse_ann} \\\n\t'  # always use init coarse ann point to limit
                f'{work_dir}/latest_results.bbox.json \\\n\t'
                f'{cfg.train_img_root} \\\n\t'
                f"{cfg.output_ann} \\\n\t"
                f"--show 0 --max-grow-rate={max_grow_rate}",
            ])
            from huicv.interactivate.path_utils import makedirs_if_not_exist
            makedirs_if_not_exist(work_dir)
            cfg.save_as_json(f'{work_dir}/coarse_config.json')

        # for i, step in enumerate(self.steps):
        #     if ('tools/dist_train.sh' in step or 'tools/train.py' in step) and ('resume-from' not in step):
        #         self.steps[i] = f'python exp/tools/fail_restart.py --max-try {self.nan_restart_time} --try-signal 256 \\\n'\
        #                         f'\'{step}\''

    def print_cmd(self):
        for i, step in enumerate(self.steps):
            print(f"# [cmd {i}:] ------------------------------------------------")
            print(step)
            print()

    def train(self, start):
        if len(self.steps) == 0:
            self.build_cmd()
        for i, step in enumerate(self.steps):
            if i < start:
                continue
            print(f"\033[1;32m[exec cmd {i}:]\033[0m", step)
            res = os.system(step)
            if res != 0 and self.stop_if_err:
                print(f"\033[1;31m[error while exec cmd {i}:]\033[0m", step)
                break


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("config", help="coarse point config file", default="TinyCOCO_Rp_Rp_Rp")
    parser.add_argument("--run", help="coarse point config file", action="store_true")
    parser.add_argument("--start", help="coarse point config file", default=0, type=int)
    args = parser.parse_args()

    # # TinyPerson
    # from exp.locpoint.coarse_config.TinyPerson_Rp_Rp_Rp import config_class, exps_config
    # # TinyCOCO
    # from exp.locpoint.coarse_config.TinyCOCO_Rp_Rp_Rp import config_class, exps_config

    _, config = os.path.split(args.config)
    config, _ = os.path.splitext(config)
    exec(f"from exp.locpoint.coarse_config.{config} import config_class, exps_config")
    # run
    exp_group = CorasePointExp(config_class, exps_config)
    if not args.run:
        exp_group.build_cmd()
        exp_group.print_cmd()
    else:
        exp_group.train(args.start)
