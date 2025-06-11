import multiprocessing
from multiprocplus import AsyncCaller
import time


# def work():
#     # import numpy as np
#     # a = np.random.random((10000, 10000))
#     # (a * a).sum()
#     time.sleep(1)


def produce_food():
    for i in range(5):
        time.sleep(1)
        pid = multiprocessing.process.current_process().pid
        print(f"produce food in {i, pid}")
        yield i, pid


def eat_food(i, pid):
    print(f"eat food from {i, pid}")
    time.sleep(1)


if __name__ == "__main__":
    # tic = time.time()
    # work()
    # print(time.time() - tic)

    print("single times consumer")
    eat_food_async = AsyncCaller(eat_food)
    tic = time.time()
    for i, pid in produce_food():
        print(i, pid)
        eat_food_async(i, pid)
    eat_food_async.join()
    print(time.time() - tic)

    print("multiple times consumer")
    eat_food_async_list = [AsyncCaller(eat_food) for _ in range(4)]
    tic = time.time()
    for i, pid in produce_food():
        print(i, pid)
        for eat_food_async in eat_food_async_list:
            eat_food_async(i, pid)
    for eat_food_async in eat_food_async_list:
        eat_food_async.join()
    print(time.time() - tic)
