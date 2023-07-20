import sys
import torch
import argparse
import os
from huicv.interactivate.path_utils import makedirs_if_not_exist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth", help="pth of .pth file")
    parser.add_argument('--save-pth', default="")
    args = parser.parse_args()

    d = torch.load(args.pth)
    keys = list(d.keys())
    for k in keys:
        if k != "state_dict":
            del d[k]

    a, b = os.path.split(args.pth)
    if len(args.save_pth) == 0:
        save_pth = os.path.join(a, f'weight_{b}')
    else:
        save_pth = args.save_pth
    makedirs_if_not_exist(save_pth)
    torch.save(d, save_pth)
    print(f"save {save_pth}.")
