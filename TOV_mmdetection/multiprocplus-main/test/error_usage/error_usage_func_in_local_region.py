

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


"""
> OUTPUT:
Process SpawnPoolWorker-2:
Traceback (most recent call last):
  File "multiprocessing\process.py", line 315, in _bootstrap
  File "multiprocessing\process.py", line 108, in run
  File "multiprocessing\pool.py", line 114, in worker
  File "multiprocessing\queues.py", line 368, in get
AttributeError: Can't get attribute 'func' on <module '__mp_main__' from 'D:\\hui\\code\\multiprocplus\\test\\error_usage\\error_usage_func_in_local_region.py'>
"""
