import sys
import multiprocessing
import threading


class FuncExcWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            self.func(*args, **kwargs)
        except BaseException as b:
            print(f"[AsyncCaller] {b}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise b


class AsyncCaller(object):
    """
    main thread:
        |
    async_func = AsyncCaller(func) ---------------------- help worker (to run the consumer loop)
                                                            |
                                            run_in_new_thread or process:
                                                data = q.get()
                                                while data is not None:
                                                    func(*data)  # can run in new process---------- task worker (run each consumer)
                                          |-------- data = q.get()
    async_func(*args): q.put(args) -------|

    """
    def __init__(self, func, help_worker_type='process', task_worker_type='process', num_task_worker=None):
        self.func = FuncExcWrapper(func)
        help_worker_type = help_worker_type.lower()
        self.task_worker_type = task_worker_type.lower()
        if help_worker_type == 'thread':
            # if help_worker run in new thread, the task_work must run in process
            self.num_task_worker = 1 if num_task_worker is None else num_task_worker
            from queue import Queue
            self.q = Queue()
            self.help_worker = threading.Thread(target=self.main_loop, args=())
        elif help_worker_type == 'process':
            # if help_worker run in new process, the task_work can run in same process as help_work
            self.num_task_worker = 0 if num_task_worker is None else num_task_worker
            self.q = multiprocessing.Queue()
            self.help_worker = multiprocessing.Process(target=self.main_loop, args=())
        else:
            raise ValueError("help_worker_type must be 'thread' or 'process'")
        self.help_worker.start()

    def __call__(self, *args, **kwargs):
        self.q.put((args, kwargs))

    def main_loop(self):
        if self.task_worker_type == 'none' or self.num_task_worker == 0:
            data = self.q.get()
            while data is not None:
                args, kwargs = data
                self.func(*args, **kwargs)
                data = self.q.get()
        elif self.task_worker_type == 'process':
            try:
                task_worker_pool = multiprocessing.Pool(self.num_task_worker)
            except ImportError as e:
                print("[Warning]: this error may caused by create too many sub threads, "
                      "Try call more AsyncCaller(your_func, help_worker_type='process') "
                      "instead of AsyncCaller(your_func, help_worker_type='thread') ")
                raise e
            try:
                data = self.q.get()
                while data is not None:
                    args, kwargs = data
                    task_worker_pool.apply_async(self.func, args=args, kwds=kwargs)
                    data = self.q.get()
            finally:
                task_worker_pool.close()
                task_worker_pool.join()
        else:
            raise ValueError()

    def join(self):
        self.q.put(None)
        self.help_worker.join()
