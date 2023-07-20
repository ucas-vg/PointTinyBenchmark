import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bash_script", default='exp/cur_exp.sh')
args = parser.parse_args()

f = open(args.bash_script, 'r')
s = f.read().replace('\r', '')
f.close()
f = open(args.bash_script, "w")
f.write(s)
f.close()

os.system(f"bash {args.bash_script}")
