from multiprocplus import multiprocess_for, MultiprocessRunner


def example_of_num_process(A, B):
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


def func(a, b):
    return a * b


if __name__ == "__main__":
    """
        - The definition of function passed to new process must be out of 'if __name__ == "__main__"' (global function or member function of global class);
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

    print("example_of_num_process")
    example_of_num_process(A, B)
