import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()

map_dir = "../pub/TOV_mmdetection/"

for f in args.files:
    shutil.copy(f, map_dir + f)
    os.system(f"cd {map_dir} && git add {f} && git commit -m \"normal update\" && git push")
