python setup.py build develop

ipython
```py
from mini_maskrcnn_benchmark import _C
print(_C.nms)
```

## normal issues

### libc10.so
```
libc10.so
----> 1 from mini_maskrcnn_benchmark import _C
ImportError: libc10.so: cannot open shared object file: No such file or directory
```
```
export LD_LIBRARY_PATH=~/.conda/envs/mmdetection/lib/python3.7/site-packages/torch/lib/:$LD_LIBRARY_PATH
```