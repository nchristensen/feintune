from dataclasses import dataclass
from typing import Union, Optional
from generators import get_trans_list
from run_tests import generic_test, run_single_param_set_v2
from autotune import TuningProblem
from autotune.space import *
from skopt.space import Real
from ytopt.search.ambs import AMBS
from frozendict import frozendict
#from ytopt.search.async_search import AsyncSearch

import hjson
import pyopencl as cl
import numpy as np
import os
import loopy as lp
from os.path import exists
from utils import convert, load_hjson, dump_hjson
from hashlib import md5
from random import shuffle


queue = None
def set_queue(pe_num, platform_num):
    global queue
    if queue is not None:
        raise ValueError("queue is already set")
    platforms = cl.get_platforms()
    # This is semi-broken because we aren't assured the device list is always
    # ordered the same, needs to order the devices by some pci_id first
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

def get_queue(pe_num, platform_num):
    #global queue
    #if queue is not None:
    #    raise ValueError("queue is already set")

    import pyopencl as cl
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue



def get_test_id(tlist):
    return md5(str(tlist).encode()).hexdigest()


def test(args):
    global queue
    print(args)
    #timeout, ((cur_test, total_tests,), (test_id, platform_id, knl, tlist, test_fn, max_flop_rate, device_latency, device_memory_bandwidth,),eval_str) = args
    #comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
    # From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD (this probably doesn't matter though. using COMM_WORLD should prevent
    # re-use of any GPU?
    
    eval_str = args["eval_str"]
    platform_id = args["platform_id"]
    #print("EVAL STR:", eval_str)

    if queue is None:
        print("Queue is none. Initializing queue")
        if eval_str == "mpi_comm_executor" or eval_str == "mpi_pool_executor":
            import mpi4py.MPI as MPI
            comm = MPI.COMM_WORLD
            set_queue(comm.Get_rank(), platform_id)
        elif eval_str == "charm4py_pool_executor":
            from charm4py import charm
            set_queue(charm.myPe(), platform_id)
        elif eval_str == "processpool":
            from os import get_pid
            set_queue(get_pid(), platform_id)
        elif eval_str == "threadpool":
            from threading import get_native_id
            set_queue(get_native_id(), platform_id)
        #elif eval_str == "ray":
        # 
        else:    
            set_queue(0, platform_id)
        
        assert queue is not None

    cur_test = args["cur_test"]
    total_tests = args["total_tests"]

    print(f"\nExecuting test {cur_test} of {total_tests}\n")
    result = run_single_param_set_v2(queue, args["knl"], args["tlist"], args["test_fn"],
            max_flop_rate=args["max_flop_rate"],
            device_memory_bandwidth=args["device_memory_bandwidth"],
            device_latency=args["device_latency"],
            timeout=args["timeout"])

    return args["test_id"], result

numeric_type = Union[np.number, float, int]

@dataclass
class ObjectiveFunction(object):

    knl: object
    eval_str: Optional[str] = None
    platform_id: int = 0
    max_flop_rate: numeric_type = np.inf
    device_memory_bandwidth: numeric_type = np.inf
    device_latency: numeric_type = 0
    timeout: Optional[numeric_type] = None

    @property
    def __name__(self):
        return self.__repr__()

    def __call__(self, p):
        params = (p["batch_size"],
                  p["kio"]*p["kii"],
                  p["kii"],
                  p["iio"]*p["iii"],
                  p["iii"],
                  p["ji"],)
        tlist = get_trans_list(self.knl, params)
        test_id = get_test_id(tlist)

        print("BEGINNING TEST")
        #args = (self.timeout, ((None, None,), (test_id, self.platform_id, self.knl, tlist,
        #        generic_test, self.max_flop_rate, self.device_latency, self.device_memory_bandwidth,),),self.eval_str,)
        args = frozendict({"timeout": self.timeout,
                "cur_test": None,
                "total_tests": None,
                "test_id": test_id,
                "platform_id": self.platform_id,
                "knl": self.knl,
                "tlist": tlist,
                "test_fn": generic_test,
                "max_flop_rate": self.max_flop_rate,
                "device_latency": self.device_latency,
                "device_memory_bandwidth": self.device_memory_bandwidth,
                "eval_str": self.eval_str
                })

        test_id, result = test(args)

        print("ENDING TEST")
        #results = run_single_param_set_v2(queue, knl, tlist, max_flop_rate=max_flop_rate,
        #            device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency, timeout=timeout)
        
        return result["data"]["avg_time"]




def offline_tuning(knl, platform_id, input_space, max_flop_rate=np.inf, device_memory_bandwidth=np.inf,
                     device_latency=0, timeout=None):

    print(input_space)
    output_space = Space([Real(0.0, inf, name="avg_time")])
    #eval_str = "mpi_comm_executor"
    #eval_str = "mpi_pool_executor"
    eval_str = "charm4py_pool_executor"
    #eval_str = "threadpool"
    #eval_str = "processpool"
    #eval_str = "ray"

    obj_func = ObjectiveFunction(knl, eval_str=eval_str, platform_id=platform_id, max_flop_rate=max_flop_rate,
                                    device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency,
                                    timeout=timeout)

    at_problem = TuningProblem(
        task_space=None,
        input_space=input_space,
        output_space=output_space,
        objective=obj_func,
        constraints=None,
        model=None)

    #for entry in range(10):
    #    p = input_space.sample_configuration(1)
    #    print(p)
    #    print(obj_func(p))
     
    # Not quite certain what the difference is between
    # these but AsyncSearch seems to support MPI. Can't find NeuralNetworksDropoutRegressor
    # or any of the other dependencies. Why is this included?
    #searcher = AsyncSearch(problem=at_problem, evaluator="ray")
    #from mpi4py import MPI
    #comm = MPI.COMM_WORLD
    searcher = AMBS(problem=at_problem, evaluator=eval_str)
    searcher.main()
