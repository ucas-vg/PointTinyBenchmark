import os

"""
1. 从work_dir中读取最新的log,解析出 loss
2. 如果loss nan了, kill当前进程
"""

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("cmd")
parser.add_argument("--max-try", type=int, default=-1)
parser.add_argument("--try-signal", type=int, default=256)  # 256 is mean train.py run got error
args = parser.parse_args()

res = 0
i = 0
while args.max_try == -1 or i < args.max_try:
    print(f"[run time {i}:] {args.cmd}")
    res = os.system(f"{args.cmd}")
    if res == args.try_signal:
        i += 1
        continue
    else:
        exit(res)
