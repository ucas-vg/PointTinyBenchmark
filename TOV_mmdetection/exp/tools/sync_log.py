import os
import sys


class LogFile(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.mtime = os.path.getmtime(filepath)

    def late_than(self, other_log_file):
        return self.mtime > other_log_file.mtime


def get_max_epoch_pth(pdir):
    max_epoch = -1
    for f in os.listdir(pdir):
        file = os.path.join(pdir, f)
        if os.path.isdir(file):
            continue
        if f.endswith('.pth') and f != 'latest.pth':
            epoch = int(f[len('epoch_'):-len('.pth')])
            if epoch > max_epoch:
                max_epoch = epoch
    return max_epoch


def get_latest_log(pdir):
    latest_log_file = None
    for f in os.listdir(pdir):
        file = os.path.join(pdir, f)
        if os.path.isdir(file):
            continue
        if f.endswith('.log'):
            log_file = LogFile(file)
            if latest_log_file is None or log_file.late_than(latest_log_file):
                latest_log_file = log_file
    return latest_log_file


def get_log(parent_dir):
    keep_log_files = []

    latest_log_file = get_latest_log(parent_dir)
    if latest_log_file is not None:
        keep_log_files.append(latest_log_file)

    for f in os.listdir(parent_dir):
        file = os.path.join(parent_dir, f)
        if os.path.isdir(file):
            keep_log_files.extend(get_log(file))
    return keep_log_files


def copy_log(log_root, save_log_root, force_cp):
    cwd = os.path.abspath(os.getcwd())
    os.chdir(log_root)
    log_files = get_log('.')
    os.chdir(cwd)

    for log_file in log_files:
        src_file = os.path.join(log_root, log_file.filepath)
        dst_file = os.path.join(save_log_root, log_file.filepath)
        if (not force_cp) and os.path.exists(dst_file):
            continue

        log_dir, log_name = os.path.split(log_file.filepath)
        save_log_dir = os.path.join(save_log_root, log_dir)
        if not os.path.exists(save_log_dir):
            os.makedirs(save_log_dir)
            print("mkdirs", log_dir)
        cmd = "cp {} {}".format(src_file, dst_file)
        print(cmd)
        os.system(cmd)


def get_invalid_save_dir(parent_dir, least_epoch=12):
    invalid_dirs = []
    for f in os.listdir(parent_dir):
        file = os.path.join(parent_dir, f)
        if os.path.isdir(file):
            invalid_dirs.extend(get_invalid_save_dir(file))

    have_sub_dir = False
    have_log_file = False
    for f in os.listdir(parent_dir):
        file = os.path.join(parent_dir, f)
        if os.path.isdir(file):
            have_sub_dir = True
        elif file.endswith('.log'):
            have_log_file = True

    max_epoch = get_max_epoch_pth(parent_dir)

    if (max_epoch < least_epoch) and (not have_sub_dir) and have_log_file:
        invalid_dirs.append(parent_dir)

    return invalid_dirs


def del_dirs(dirs):
    for d in dirs:
        cmd = "rm -rf {}".format(d)
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    class Args(object):
        force_cp = False

    args = Args()
    if len(sys.argv) > 1:
        if sys.argv[1] == '-f':
            args.force_cp = True
        else:
            raise ValueError("args[1] must be -f or not given")
    log_root = "../TOV_mmdetection_cache/work_dir"
    save_log_root = './log'

    del_dirs(get_invalid_save_dir(log_root, least_epoch=12))
    print()
    copy_log(log_root, save_log_root, args.force_cp)
