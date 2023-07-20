import os


cfg_options = dict(
    evaluation=dict(
        interval=1, metric='bbox',
        use_location_metric=True,
        location_kwargs=dict(
            class_wise=False,
            matcher_kwargs=dict(multi_match_not_false_alarm=False),
            location_param=dict(
                matchThs=[0.5, 1.0, 2.0],
                # recThrs='np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                maxDets=[300],
                # recThrs='np.linspace(.90, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)',
                # maxDets=[1000],
            )
        )
    )
)


def format_option(cfg_dict, prefix=""):
    options = []
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            options.extend(format_option(v, f'{prefix}{k}.'))
        else:
            if isinstance(v, list):
                v = f"'{v}'".replace(' ', '')
            elif isinstance(v, str):
                v = f"'{v}'"
            options.append(f"{prefix}{k}={v}")
    return options


class Arg(object):
    def __init__(self, arg_name, arg_idx, start_idx, end_idx):
        self.arg_name = arg_name
        self.arg_idx = arg_idx
        self.start_idx = start_idx
        self.end_idx = end_idx


class CMD(object):
    def __init__(self, cmd_str):
        self.cmd = cmd_str

    def find_arg(self, arg_name):
        """
            ${arg_name} ${arg_value}
        """
        cmd = self.cmd
        idx = cmd.find(arg_name)
        if idx == -1:
            return None
        arg_idx = idx

        idx += len(arg_name)
        assert cmd[idx].isspace(), cmd[idx:]

        while cmd[idx].isspace():
            idx += 1
            if idx >= len(cmd):
                return None
        start_idx = idx

        while not cmd[idx].isspace():
            idx += 1
            if idx >= len(cmd):
                break
        end_idx = idx
        return Arg(arg_name, arg_idx, start_idx, end_idx)

    def replace_arg(self, arg_name, value):
        if isinstance(arg_name, str):
            arg = self.find_arg(arg_name)
            if arg is None:
                self.add_arg(arg_name, value)
                return
        elif isinstance(arg_name, Arg):
            arg = arg_name
        else:
            raise TypeError("")
        self.cmd = f"{self.cmd[:arg.start_idx]} {value}{self.cmd[arg.end_idx:]}"

    def append_arg(self, arg_name, value):
        if isinstance(arg_name, str):
            arg = self.find_arg(arg_name)
            if arg is None:
                self.add_arg(arg_name, value)
                return
        elif isinstance(arg_name, Arg):
            arg = arg_name
        else:
            raise TypeError("")

        self.cmd = f"{self.cmd[:arg.end_idx]}{value}{self.cmd[arg.end_idx:]}"

    def add_arg(self, arg_name, value):
        self.cmd += f' {arg_name} {value}'

    def get_value(self, arg):
        return self.cmd[arg.start_idx:arg.end_idx]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_cmd")
    args = parser.parse_args()

    train_cmd = CMD(args.train_cmd)
    arg = train_cmd.find_arg('--work-dir')
    assert arg is not None, '--work-dir must be specified'
    work_dir = train_cmd.get_value(arg)
    train_cmd.replace_arg(arg, os.path.join(work_dir, 'loc_eval'))

    cfg_options = " evaluation.do_final_eval=True  " + " ".join(format_option(cfg_options))
    train_cmd.append_arg('--cfg-options', cfg_options)

    train_cmd.add_arg('--resume-from', f'{work_dir}/latest.pth')

    localization_eval_cmd = train_cmd
    print("\n[cmd]:")
    print(localization_eval_cmd.cmd)
    print()
    os.system(localization_eval_cmd.cmd)
