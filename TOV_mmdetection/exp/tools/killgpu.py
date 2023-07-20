#!/usr/bin/python3

import os
import argparse


def parse_nvidia_smi():
    info = os.popen("nvidia-smi")

    datas = []
    for line in info.readlines():
        line = line.strip()
        if line.startswith("|    ") and line.endswith("MiB |"):
            line_data = [e for e in line.split(' ') if len(e) > 0]
            if len(line_data) not in [7, 9]:
                print("warning", line_data, line)
                continue
            gpu_id = int(line_data[1])
            pid = int(line_data[-5])
            datas.append({
                "gpu_id": gpu_id,
                "pid": pid,
                'type': line_data[-4],
                "name": line_data[-3],
                "mem": line_data[-2],
            })
    return datas


def parse_seq(s, type=int, range=range):
    seq = []
    for e in s.split(','):
        e = e.split('-')
        if len(e) == 1:
            seq.append(type(e[0]))
        elif len(e) == 2:
            seq.extend(list(range(type(e[0]), type(e[1])+1)))
        else:
            raise ValueError(e)
    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser("\n\tpython kill.py 0-7\n\tpython kill.py 0,3,7\n")
    parser.add_argument("kill_gpus", help="killed gpu ids, such as 0,1,3 or 0-5")
    args = parser.parse_args()

    kill_gpus = parse_seq(args.kill_gpus)

    datas = parse_nvidia_smi()
    pids = []
    for data in datas:
        if data["gpu_id"] in kill_gpus and data['name'].find('python') != -1 and data['type'] == 'C':
            pids.append(str(data["pid"]))

    if len(pids) > 0:
        cmd = f"kill {' '.join(pids)}"
        print(cmd)
        os.system(cmd)
