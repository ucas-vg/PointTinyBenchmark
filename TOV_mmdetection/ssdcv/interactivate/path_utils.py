import os
import os.path as osp
from .cmd import input_yes


def makedirs_if_not_exist(file_path):
    if osp.isdir(file_path):
        if not osp.exists(file_path):
            os.makedirs(file_path)
            return True
    else:
        file_dir, file_name = osp.split(file_path)
        if not osp.exists(file_dir):
            os.makedirs(file_dir)
            return True
    return False


def override_check(file_path):
    if os.path.exists(file_path):
        return input_yes(f"have exists: {file_path}\ndo you want to overwrite?(yes/no)")
    return True
