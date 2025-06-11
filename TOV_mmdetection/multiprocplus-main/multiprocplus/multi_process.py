import multiprocessing
import sys
import time
import os

__all__ = ["MultiprocessRunner", "MultiprocessFor", "multiprocess_for", "mp_for"]


class Task(object):
    def __init__(self, idx, func, args, cost):
        self.idx = idx
        self.func = func
        self.args = args
        self.cost = cost

    def __str__(self):
        return f"<T{self.idx}: cost={self.cost}>"

    def __repr__(self):
        return self.__str__()


class TypeConvert(object):
    """
        because manager can easy to solve list and dict with ListProxy and DictProxy
        so given a non-basic type (int, float, str, list, dict), we convert it to list by add a _fake_list_flag,
        and get_var to turn it back while passed to func that called in new process.
    """

    @staticmethod
    def get_var(var_list):
        vars = []
        for data in var_list:
            if isinstance(data, multiprocessing.managers.ListProxy) and \
                    len(data) == 2 and isinstance(data[0], TypeConvert):  # see line47
                data = data[1]
            vars.append(data)
        return tuple(vars)

    @staticmethod
    def to_shared_memory_var(manager, var_list):
        m_share_data_list = []
        import numpy as np
        for data in var_list:
            if isinstance(data, list):
                data = manager.list(data)
            elif isinstance(data, dict):
                data = manager.dict(data)
            elif isinstance(data, (int, float)):
                pass
            else:
                data = manager.list([_fake_list_flag, data])
                # raise TypeError("shared data must be list or dict")
            m_share_data_list.append(data)
        return tuple(m_share_data_list)


_fake_list_flag = TypeConvert()


class MultiprocessRunner(object):
    """
    param:
     num_process: 0, 1 will not run in multi process but using for, -1 will using all cpu to run.
    we advise set num_process=0/1 while debug to make sure passed func have no problem.

    There are some problems that have no warning and errors, but fail to execute  while in multiprocess solver:
        1. passed func declaration mismatch with args
        2. passed func have bugs
        3. passed func must be a global function or member function of global class.

    manager.list()/dict(), the var passed to new process must keep ref in cur process, or will got access error.

    cost assign works well while shared data is big.

    addition:
     1) pool.apply_async do not contains any time of copy passed args and running passed func, all these time are in new process.
     2) the same passed args to two different process will cost (two * copy time), even same args will copy twice.
        so we should only pass process-specified args to each process.
     3)
    """

    def __init__(self, num_process=multiprocessing.cpu_count(),
                 sync_obj_args=False, cost_rate_per_process=1.0,
                 debug_info=0):
        """
            sync_obj_args: whether use manager to manage object argument,
                while set True, the args modified in sub process will synchronization to main process.
            cost_rate_per_process:
        """
        if isinstance(num_process, int):
            if num_process < 0:
                num_process = multiprocessing.cpu_count()
        elif isinstance(num_process, float):
            assert 0.0 <= num_process <= 1.0, f"num_process must be float in [0.0, 1.0] or int," \
                                              f" but got <{type(num_process)} {num_process}>"
            num_process = int(round(num_process * multiprocessing.cpu_count()))
        else:
            raise ValueError(f"num_process must be float in [0.0, 1.0] or int, but got <{type(num_process)} {num_process}>")
        self.num_process = min(num_process, multiprocessing.cpu_count())
        # self.pool = multiprocessing.Pool(self.num_process)
        self.sync_obj_args = sync_obj_args
        self.debug_info = debug_info
        self.cost_rate_per_process = cost_rate_per_process

    def __call__(self, func, args_list, share_data_list, cost_list=None):
        if self.debug_info > 0:
            tic = time.time()
        self.func = func
        if self.num_process == 1 or self.num_process == 0:
            if self.debug_info > 0:
                print(f"[MultiprocessRunner] start run in single process for {len(args_list)} tasks")
            results = []
            for i, args in enumerate(args_list):
                args = args + tuple(share_data_list)
                results.append(func(*args))
            return results
        else:
            if cost_list is not None:
                results = self.run_with_cost(func, args_list, share_data_list, cost_list)
            else:
                results = self.run(func, args_list, share_data_list)
        if self.debug_info > 0:
            print(f"[MultiprocessRunner] finished all tasks cost {time.time()-tic}s")
        return results

    def run(self, func, args_list, share_data_list=[]):
        num_process = min(self.num_process, len(args_list))
        # will not delete shared memory after return compare to with multiprocessing.Manager() as manager
        manager = multiprocessing.Manager()
        pool = multiprocessing.Pool(num_process)

        if self.debug_info > 0:
            print(f"[MultiprocessRunner] start run async in {num_process} processes for {len(args_list)} tasks")
        # results = manager.dict()
        results = [manager.dict() for _ in range(len(args_list))]
        if self.sync_obj_args:
            share_data_list = TypeConvert.to_shared_memory_var(manager, share_data_list)
            args_list = [TypeConvert.to_shared_memory_var(manager, args) for i, args in enumerate(args_list)]
        for i, args in enumerate(args_list):  # the loop not contains time of copy args and running
            # print(f"{i}-th process:", time.time())
            args = (i, results[i],) + tuple(args) + tuple(share_data_list)
            # print(f"start {i}-th process.")
            pool.apply_async(self.func_wrapper, args=args)  # generate new process, new process copy args firstly.
        pool.close()
        pool.join()

        return self.dicts_to_list(results)

    def run_with_cost(self, func, args_list, share_data_list=[], cost_list=[]):
        assert len(args_list) == len(cost_list), f"{len(args_list)} vs {len(cost_list)}"
        tasks_group = self.assign_task_group(func, args_list, cost_list)
        task_group_ids = [[task.idx for task in tasks] for tasks in tasks_group]
        task_group_args = [[task.args for task in tasks] for tasks in tasks_group]
        if self.debug_info > 1:
            print(f"assigned task of each processes: {task_group_ids}")

        num_process = min(self.num_process, len(tasks_group))
        # will not delete shared memory after return compare to with multiprocessing.Manager() as manager
        manager = multiprocessing.Manager()
        pool = multiprocessing.Pool(num_process)

        if self.debug_info > 0:
            print(f"[MultiprocessRunner] start run async in {num_process} processes "
                  f"for {len(tasks_group)} groups ({len(args_list)} tasks)")
        # the manger var must keep ref during new process running
        # results = manager.dict()
        results = [manager.dict() for _ in range(len(tasks_group))]
        if self.sync_obj_args:
            share_data_list = TypeConvert.to_shared_memory_var(manager, share_data_list)
            task_group_args = TypeConvert.to_shared_memory_var(manager, task_group_args)
        for gid, tasks in enumerate(tasks_group):
            args = (gid, task_group_ids[gid], results[gid]) + (task_group_args[gid], share_data_list)
            pool.apply_async(self.multi_func_wrapper, args=args)  # generate new process, new process copy args firstly.
        pool.close()
        pool.join()

        return self.dicts_to_list(results)

    def multi_func_wrapper(self, gid, tasks_id, results, multi_args, share_data_list):
        try:
            tic = time.time()
            for i, task_id in enumerate(tasks_id):
                tic2 = time.time()
                self.func_wrapper(task_id, results, *multi_args[i], *share_data_list)
                # print(f"[MultiprocessRunner:({gid})] complete {task_id}-th task ({i + 1}/{len(tasks_id)}, {time.time()-tic2}s).")
            if self.debug_info > 0:
                print(f"[MultiprocessRunner:({gid})] finished all {len(tasks_id)} tasks "
                      f"({time.time() - tic}s) --------------------------")
        except BaseException as b:
            print(f"[MultiprocessRunner:({gid})] {b}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise b

    def func_wrapper(self, i, results, *args, **kwargs):
        try:
            args = TypeConvert.get_var(args)
            results[i] = self.func(*args, **kwargs)
        except BaseException as b:
            print(f"[MultiprocessRunner:({i})] {b}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise b

    def assign_task_group(self, func, args_list, cost_list):
        tasks = [Task(i, func, args, cost) for i, (args, cost) in enumerate(zip(args_list, cost_list))]
        tasks = sorted(tasks, key=lambda task: -task.cost)
        std_cost = tasks[0].cost * self.cost_rate_per_process
        tasks_gorup = []
        cur_cost, cur_group = 0, []
        for i in range(len(tasks) - 1, -1, -1):
            cur_cost += tasks[i].cost
            if cur_cost <= std_cost:
                cur_group.append(tasks[i])
            else:
                if cur_cost - std_cost <= std_cost - (cur_cost - tasks[i].cost):  # add cur task is better than not add
                    cur_group.append(tasks[i])
                    tasks_gorup.append(cur_group)
                    cur_cost, cur_group = 0, []
                else:  # not add is beeter than add, keep cur task to next group
                    tasks_gorup.append(cur_group)
                    cur_cost, cur_group = tasks[i].cost, [tasks[i]]
        if len(cur_group) > 0:
            tasks_gorup.append(cur_group)
        return tasks_gorup

    def dicts_to_list(self, results):
        merged_results = {}
        [merged_results.update(result) for result in results]

        new_data = []
        for idx in range(len(merged_results)):
            new_data.append(merged_results[idx])
        return new_data


def multiprocess_for(func, args_list, share_data_list=[], cost_list=None,
                     num_process=multiprocessing.cpu_count(), debug_info=False, **kwargs):
    """
    multiprocess_for can be utilized to replace for loop:
        results = []
        for args in args_list:
            args = args + (share_data_list,)
            results.append(func(*args))
        return results
    """
    return MultiprocessRunner(num_process, debug_info=debug_info, **kwargs)(func, args_list, share_data_list, cost_list)


MultiprocessFor = MultiprocessRunner
mp_for = multiprocess_for
