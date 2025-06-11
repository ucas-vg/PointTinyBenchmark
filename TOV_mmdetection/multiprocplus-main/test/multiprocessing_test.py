import multiprocessing
import time


"""
- new process generate will copy data and code out of "if __name__ == '__main__':" of entry script 
    and run the passed function as entry script.
- the copy data changed in new process will not update to other process, manager is used to sync passed args between 
    origin process and new process. Beside, if same manager var is passed to multi new processes, manager will sync 
    the var for all processes.
"""


def f(i, norm_list, manager_list, n):
    time.sleep(3)
    norm_list.append(n)
    manager_list.append(n)

    print(f"{i}-th process, normal_list", norm_list)
    print(f"{i}-th process, manager_list", manager_list)


if __name__ == '__main__':
    m = multiprocessing.Manager()
    p = multiprocessing.Pool(3)
    a_list = list(range(4))
    m_dict = m.dict()
    m_list = m.list(a_list)
    p_list = []
    for i in range(3):
        # the args passed to new process is simple copy if not variable of manager
        p.apply_async(f, args=(i, a_list, m_list, i))
    time.sleep(2)  # make sure sub process have created and run. after sub process run, modify a_list and m_list
    a_list.append("a")
    m_list.append("a")
    p.close()
    p.join()
    print(a_list)
    print(m_list)
