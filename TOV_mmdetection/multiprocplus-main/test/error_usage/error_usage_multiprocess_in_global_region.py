"""
> Bug explain:
1. multiprocess_run must be called in a function called under 'if __name__ == "__main__"'
    
> Error:
    RuntimeError: 
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase. 

> Fix:
 see test/multiprocplus_test.py:
"""

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


"""
> OUTPUT:
328350
328350
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
"""
