import numpy as np
from multiprocplus import multiprocess_for, MultiprocessRunner


# class Test1(object):
#     def func(self):


class MyTest(object):
    def func(self, idx, dataA, dataB):
        print(f"start {idx}")
        da = dataA[idx]
        sum_res = {}
        for k, v in dataB.items():
            sum_res[k] = da + v
        import time
        time.sleep(3)
        print(f"finish {idx}")
        return sum_res

    def func2(self, idx, da, dataB):
        print(f"start {idx}")
        sum_res = {}
        da = da[0]
        for k, v in dataB.items():
            sum_res[k] = da + v
        import time
        time.sleep((idx + 1) * 2)
        print(f"finish {idx}")
        return sum_res
        # return idx

    def mytest1(self):
        """
        the shared data can be list or dict, even numpy data inside them.
        """
        import numpy as np
        n = 5
        dataA = [np.arange(n), np.arange(n) + 5, np.arange(n) + 10, np.arange(n) + 15, np.arange(n) + 20]  # data
        dataB = {"a": np.arange(n), "b": 2, "c": 3}

        # # for a, b in zip(dataA, dataB):
        # #     func(a, b)
        # res = MultiprocessRunner(num_process=3)(self.func, [(i,) for i in range(5)], share_data_list=[dataA, dataB])
        # print(len(res), res[1])
        # res = MultiprocessRunner(num_process=0)(self.func, [(i,) for i in range(5)], share_data_list=[dataA, dataB])
        # print(len(res), res[1])
        # # print(res)
        # # print(res[0])
        # # res = multiprocess_run(self.func, [(i,) for i in range(5)], share_data_list=[dataA, dataB], num_process=3)
        # # print(res[1])
        #
        # res = multiprocess_run(self.func2, [(i, [dataA[i]]) for i in range(5)], share_data_list=[dataB],
        #                        num_process=3)
        # print(len(res), res[1])

        res = multiprocess_for(self.func2, [(i, [dataA[i]]) for i in range(5)], share_data_list=[dataB],
                               cost_list=[1, 2, 3, 4, 5],  # estimated cost time of each processes
                               num_process=3, debug_info=2, cost_rate_per_process=1.0)
        print(len(res))
        print(res[1])
        return res

    def mytest2(self, res):
        """
            the returned result can be passed cross function.
        """
        print(res[0])


if __name__ == "__main__":
    # res = mytest1()
    # mytest2(res)
    m = MyTest()
    m.mytest2(m.mytest1())
