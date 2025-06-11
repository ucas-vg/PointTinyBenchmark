from multiprocplus import multiprocess_for, MultiprocessRunner


def func(a, b):
    return a * b


if __name__ == "__main__":
    """
        - The definition of function passed to new process must be out of 'if __name__ == "__main__"' 
            (global function or member function of global class);
        - Code/Function that you do not want to run in new process must be written/called under 'if __name__ == "__main__"' of entry script, 
          or it will run/called in new process.
        - Following last note, multiprocess_run must be called in a function called under 'if __name__ == "__main__"' of entry script. 
          Otherwise, new processes will be generated recursively
    """
    N = 100
    A, B = list(range(N)), list(range(N))

    # => run in single process
    C = [func(a, b) for a, b in zip(A, B)]
    print(sum(C))
    # => run in 3 processes
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=3)
    print(sum(C))

    import numpy as np
    N = 100
    # A, B = list(range(N)), list(range(N))
    A, B = [np.arange(10) + _ for _ in range(N)], [np.arange(10) - _ for _ in range(N)]

    # => run in single process
    C = [func(a, b) for a, b in zip(A, B)]
    print(sum(C))
    # => run in 3 processes
    C = multiprocess_for(func, [(a, b) for a, b in zip(A, B)], num_process=10)
    print(sum(C))
