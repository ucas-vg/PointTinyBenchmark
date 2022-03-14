import os
import shutil
import sys

dirs = list(sys.argv[1:])

s, e = len('epoch_'), -len('.pth')
while len(dirs) != 0:
    the_dir = dirs.pop(0)
    epoch_fpath_map = {}
    for f in os.listdir(the_dir):
        fpath = os.path.join(the_dir, f)
        if os.path.isdir(fpath):
            dirs.append(fpath)
        else:
            if f.endswith('.pth') and f != 'latest.pth':
                try:
                    epoch = int(f[s:e])
                    epoch_fpath_map[epoch] = fpath
                except ValueError as e:
                    print(e, file=sys.stderr)
    if len(epoch_fpath_map) > 0:
        epochs = list(epoch_fpath_map.keys())
        keep_epoch = [max(epochs)]
        for epoch, fpath in epoch_fpath_map.items():
            if epoch not in keep_epoch:
                print('rm {}'.format(fpath))
                os.remove(fpath)
