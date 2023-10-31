from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
from charm4py.pool import PoolScheduler, Pool, PoolExecutor
from charm4py.charm import Charm, CharmRemote
import numpy as np
from time import sleep


def multiply(a, b):
    sleep(10*np.random.rand())
    return a*b


def main(arg):
    data = np.arange(1, 100)
    pool_proxy = Chare(PoolScheduler, onPE=0)
    executor = PoolExecutor(pool_proxy)

    # result = executor.map(multiply, data, data)
    # print(result)
    # print(list(zip(data, data, result)))

    futures = [executor.submit(multiply, entry, entry) for entry in data]

    from concurrent.futures import wait
    from time import time, sleep
    futures[0].set_running_or_notify_cancel()

    for future in futures:
        print(future._state)
        # future.set_running_or_notify_cancel()
        # future.deposit(5)
        # future.send(result=5)

    # futures[0].get()

    for future in futures:
        # print("Called GET at time", time())
        print(future.values)
        # print(future.gotvalues)
        # print(future)
        # future.cancel()
        # future()
        # print(future)
    """
    for future in futures:
        #print(future._state)
        future.set_running_or_notify_cancel()
    sleep(3)

    for future in futures:
        print(future._state)
 

    """

    # wait(futures)
    exit()


charm.start(main)
exit()  # charm.exit freezes the program
charm.exit()
