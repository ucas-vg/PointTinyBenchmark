import time
from multiprocplus import multiprocess_for, MultiprocessRunner
import numpy as np


"""
# the cost used to group is helpful while share data is large
"""


def func(i, a, b, c, S):
    time.sleep(c)
    return a * b + S.mean()


if __name__ == "__main__":
    """
    """
    N = 1000
    A, B = np.random.randn(N, 10000), np.random.randn(N, 10000)  # list(range(N)), list(range(N))
    S = np.random.randn(100, 100)
    cost = np.random.uniform(0, 0.1, (N-20,)).tolist() + np.random.uniform(3, 5, (20,)).tolist()

    # => run in 10 processes
    tic = time.time()
    C = multiprocess_for(func, [(i, a, b, c) for i, (a, b, c) in enumerate(zip(A, B, cost))], share_data_list=[S],
                         num_process=6, debug_info=2)
    print("result:", np.sum(C), "cost time:", time.time() - tic)

    tic = time.time()
    C = multiprocess_for(func, [(i, a, b, c) for i, (a, b, c) in enumerate(zip(A, B, cost))], share_data_list=[S],
                         cost_list=cost, num_process=6, debug_info=2)
    print("result:", np.sum(C), "cost time:", time.time() - tic)

    # => run in single process
    tic = time.time()
    C = [func(i, a, b, c, S) for i, (a, b, c) in enumerate(zip(A, B, cost))]
    print("result:", np.sum(C), "cost time:", time.time() - tic)

