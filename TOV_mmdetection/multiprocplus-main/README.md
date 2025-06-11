# multiprocplus
multiprocessing plus, a drop-in and easy-use enhancement for multiprcocessing

- easy to replace "for loop".
- assign task group with given an estimated cost.

# install

```shell
# install method 1: from source code
git clone https://github.com/yinglang/multiprocplus
cd multiprocplus
python setup.py install
# install method 2: from pip
pip install multiprocplus
```

# Usage

```shell
from multiprocplus import multiprocess_for, MultiprocessRunner

# func passed to new process must be global function or 
# member function of global class, and not lambda function
def func(a, b):
    return a * b

if __name__ == "__main__":
    N = 100
    A, B = list(range(N)), list(range(N))
    # run in single process
    C = [func(a, b) for a, b in zip(A, B)]
    print(sum(C))
    # => run in multiprocess
    # only allow to be called under 'if __name__ == "__main__"' of entry script
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
    print(sum(C))
    
    # run 10 samples as a gorup for each sub-process
    cost_list = [1] * len(A)
    cost_list[0] = 10                             # 10 samples
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3, cost_list=cost_list)
    # open fully debug info
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3, cost_list=cost_list, debug_info=2)
```

For more usage examples, see [doc/usage.md](doc/usage.md)

# Notice:

>1. The definition of function passed to new process must be out of 'if __name__ == "__main__"' (global function or member function of global class), 
  and can not be lambda function;
>2. Code/Function that you do not want to run in new process must be written/called under 'if __name__ == "__main__"' of entry script, 
  or it will run/called in new process.
>3. Following last note, multiprocess_run must be called in a function called under 'if __name__ == "__main__"' of entry script. 
  Otherwise, new processes will be generated recursively

For error usage examples, to see [doc/normal_issue.md](doc/normal_issue.md) or [test/error_usage](test/error_usage)

# Basic && Implementation

## Manager & Pool in multiprocessing
If you have known the usage of module multiprocessing in python, you can skip this section.
see [CSDN](https://blog.csdn.net/yinglang19941010/article/details/127390585)

## Implementation
If you just want to use multiprocplus in you project and do not interest the implementation, you can skip this section.
