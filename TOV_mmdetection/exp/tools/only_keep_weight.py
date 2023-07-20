import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pth")
parser.add_argument("--save_pth", default="", type=str, help="")
args = parser.parse_args()

d = torch.load(args.pth)
keys = list(d.keys())
for key in keys:
    if key != 'state_dict':
        del d[key]

save_path = args.pth if len(args.save_pth) == 0 else args.save_pth
torch.save(d, save_path)
