
# API
### multiprocess_for
```python
def multiprocess_for(func, args_list, share_data_list=[], cost_list=None, num_process=multiprocessing.cpu_count(), debug_info=False):
```

the equal implementation of single process can be treated as:
```python
result = []
for args in args_list:
    args = args + share_data_list
    result.append(func(*args))
return result
```

#### 0. args_list

args_list is list of tuple, each tuple contains args passed to func
we only support arg

#### 1. [num_process](test/multiprocplus_test2_num_process.py)

```python
# => run in single process
print("run in single process")
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=0.0, debug_info=True)  # 0.0 * cpu_count()
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=0, debug_info=True)
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=1, debug_info=True)

# => run in all processes
print("run in all processes")
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], debug_info=True)
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=1.0, debug_info=True)   # 1.0 * cpu_count()
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=-1, debug_info=True)

# => run in half of all processes
print("run in half of all processes")
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=0.5, debug_info=True)   # 0.5 * cpu_count()

# => run in 2 processes
print("run in 2 processes")
C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=2, debug_info=True)
```

#### 2. [share_data_list](test/multiprocplus_test2_share_data.py)
```python
def example_of_shared_data(A, B):
    # => run in single process
    C = [func(a, b) for a, b in zip(A, B)]
    print(sum(C))
    # => run in 3 processes
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
    C = multiprocess_for(func1, [(idx, a) for idx, a in enumerate(A)], share_data_list=[B], num_process=3)
    print(sum(C))

def func1(idx, a, B):
    return a * B[idx]

def func(a, b):
    return a * b
```

#### 3. [cost_list](test/multiprocplus_test2_cost.py)

```python
import time
from multiprocplus import multiprocess_for, MultiprocessRunner


def func(i, a, b):
    time.sleep((i+1)/2)
    return a * b


if __name__ == "__main__":
    """
    """
    N = 5
    A, B = list(range(N)), list(range(N))
    cost = [1, 2, 3, 4, 5]

    # => run in 3 processes
    tic = time.time()
    C = multiprocess_for(func, [(i, a, b) for i, (a, b) in enumerate(zip(A, B))],
                         num_process=3, debug_info=2)
    print("result:", sum(C), "cost time:", time.time() - tic)

    tic = time.time()
    C = multiprocess_for(func, [(i, a, b) for i, (a, b) in enumerate(zip(A, B))],
                         cost_list=cost, num_process=3, debug_info=2)
    print("result:", sum(C), "cost time:", time.time() - tic)

    # => run in single process
    tic = time.time()
    C = [func(i, a, b) for i, (a, b) in enumerate(zip(A, B))]
    print("result:", sum(C), "cost time:", time.time() - tic)
```

```shell
[MultiprocessRunner] start run async in 3 processes for 5 tasks
result: 30 cost time: 3.7509748935699463
assigned task of each processes: [[0, 1, 2], [3], [4]]
[MultiprocessRunner] start run async in 3 processes for 3 groups (5 tasks)
[MultiprocessRunner:(1)] finished all 1 tasks (2.016202688217163s) --------------------------
[MultiprocessRunner:(0)] finished all 3 tasks (3.026254653930664s) --------------------------
[MultiprocessRunner:(2)] finished all 1 tasks (2.513298511505127s) --------------------------
result: 30 cost time: 3.2153007984161377
result: 30 cost time: 7.5298871994018555

Process finished with exit code 0
```

### AsyncCaller

```shell
import multiprocessing
from multiprocplus import AsyncCaller

def produce_food():
    for i in range(5):
        time.sleep(1)
        pid = multiprocessing.process.current_process().pid
        print(f"produce food in {i, pid}")
        yield i, pid

def eat_food(i, pid):
    print(f"eat food from {i, pid}")
    time.sleep(1)
    
# run in same process
for i, pid in produce_food():
    print(i, pid)
    eat_food(i, pid)

# run in async process
eat_food = AsyncCaller(eat_food)
for i, pid in produce_food():
    print(i, pid)
    eat_food(i, pid)
eat_food.join()
```
