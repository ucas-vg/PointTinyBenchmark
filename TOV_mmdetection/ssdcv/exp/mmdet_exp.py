from copy import deepcopy
import os
import argparse


def update_dict(old_d, new_d, allow_overwrite=True):
    nd = deepcopy(old_d)
    for k, v in new_d.items():
        if isinstance(v, dict) and k in old_d:
            nd[k] = update_dict(old_d[k], v, allow_overwrite)
        else:
            if not allow_overwrite and k in nd:
                raise ValueError(f"{k}={nd[k]} have been set in origin dict, but overwrite({k}={v}) in new dict.")
            nd[k] = v
    return nd


def cfg_to_str(cfg, prefix=''):
    cfg_opt = ""
    for k, v in cfg.items():
        if len(prefix) > 0:
            k = f'{prefix}.{k}'
        if isinstance(v, dict):
            cfg_opt += cfg_to_str(v, prefix=k)
        elif isinstance(v, (tuple, list)):
            raise NotImplementedError
        elif isinstance(v, (int, float, str)):
            cfg_opt += f' {k}={v}'
        else:
            raise ValueError(v)
    return cfg_opt


class BaseExp(object):
    def kill_cmd(self, gpu_str):
        cmd = f'python exp/tools/killgpu.py {gpu_str}'
        return cmd

    def get_cmds(self, cfg, g_cfg):
        raise NotImplementedError

    def run(self, cfg, g_cfg, args):
        cmds = self.get_cmds(cfg, g_cfg)
        for i, cmd in enumerate(cmds):
            if i >= args.start:
                if args.end < 0 or i < args.end:
                    print(f"# [cmd {i}]")
                    print(cmd)
                    print()
                    if args.run:
                        os.system(cmd)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", help="", action="store_true")
        parser.add_argument("--start", help="", default=0, type=int)
        parser.add_argument("--end", help="", default=-1, type=int)
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    cfg = dict(data=dict(samples_per_gpu=4), optimizer=dict(lr=0.01))
    new_cfg = dict(data=dict(samples_per_gpu=1), optimizer=dict(lr=0.01, a=1), b=2)
    cfg = update_dict(cfg, new_cfg)
    print(cfg)
    print(cfg_to_str(cfg))
