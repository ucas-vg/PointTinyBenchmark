# Error Usage

## Error code example

The no-bugs example is in [test/multiprocplus_test.py](test/multiprocplus_test.py)

#### error #1: create multiprocess out 'if __name__ == "__main__"' of entry script
- Code/Function that you do not want to run in new process must be written/called under 'if __name__ == "__main__"' of entry script, 
  or it will run/called in new process.
- Following last note, multiprocess_for must be called in a function called under 'if __name__ == "__main__"' of entry script. 
  Otherwise, new processes will be generated recursively util crashed.
  
```py
from multiprocplus import multiprocess_for, MultiprocessRunner

def func(a, b):
    return a * b

N = 100
A, B = list(range(N)), list(range(N))

# => run in single process
C = [func(a, b) for a, b in zip(A, B)]
print(sum(C))
# => run in multiprocess
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
print(sum(C))
```
it will get output/bug as:
```shell
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "multiprocessing\spawn.py", line 116, in spawn_main
      File "multiprocessing\spawn.py", line 125, in _main
      File "multiprocessing\spawn.py", line 236, in prepare
      File "multiprocessing\spawn.py", line 287, in _fixup_main_from_path
      File "runpy.py", line 268, in run_path
      File "runpy.py", line 97, in _run_module_code
      File "runpy.py", line 87, in _run_code
      File "D:\hui\code\multiprocplus\test\error_usage\error_usage_run_in_global_region.py", line 20, in <module>
        C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
      File "D:\hui\code\multiprocplus\multiprocplus\multi_process.py", line 180, in multiprocess_for
        return MultiprocessRunner(num_process)(func, args_list, share_data_list, cost_list)
      File "D:\hui\code\multiprocplus\multiprocplus\multi_process.py", line 64, in __call__
        return self.run(func, args_list, share_data_list)
      File "D:\hui\code\multiprocplus\multiprocplus\multi_process.py", line 69, in run
        manager = multiprocessing.Manager()
      File "multiprocessing\context.py", line 57, in Manager
      File "multiprocessing\managers.py", line 553, in start
      File "multiprocessing\process.py", line 121, in start
      File "multiprocessing\context.py", line 327, in _Popen
      File "multiprocessing\popen_spawn_win32.py", line 45, in __init__
      File "multiprocessing\spawn.py", line 154, in get_preparation_data
      File "multiprocessing\spawn.py", line 134, in _check_not_importing_main
    RuntimeError: 
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.
    
            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:
    
                if __name__ == '__main__':
                    freeze_support()
                    ...
    
            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.
```

#### error #2:

- Following last note, multiprocess_run must be called in a function called under 'if __name__ == "__main__"' of entry script. 
  Otherwise, new processes will be generated recursively

```python
from multiprocplus import multiprocess_for

func1 = lambda a, b: a*b

if __name__ == "__main__":
    N = 100
    A, B = list(range(N)), list(range(N))

    def func(a, b):
        return a * b

    # => run in single process
    C = [func(a, b) for a, b in zip(A, B)]
    print(sum(C))
    # => run in multiprocess
    # error usage 1:
    C = multiprocess_for(func1, [(a, b) for a, b in zip(A, B)], num_process=3)  # error but no error info report
    # error usage 2:
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
    print(sum(C))
```

if function is **lambda function**, the multiprocess_for will **failed to run, but no error info report**;
if function is not global, it will get such output/bugs as:
```shell
Process SpawnPoolWorker-2:
Traceback (most recent call last):
  File "multiprocessing\process.py", line 315, in _bootstrap
  File "multiprocessing\process.py", line 108, in run
  File "multiprocessing\pool.py", line 114, in worker
  File "multiprocessing\queues.py", line 368, in get
AttributeError: Can't get attribute 'func' on <module '__mp_main__' from 'D:\\hui\\code\\multiprocplus\\test\\error_usage\\error_usage_func_in_local_region.py'>
```

