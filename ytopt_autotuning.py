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
def set_queue(exec_id, platform_num):

    global queue
    if queue is not None:
        raise ValueError("queue is already set")
    platforms = cl.get_platforms()

    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[exec_id % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


def get_test_id(tlist):
    return md5(str(tlist).encode()).hexdigest()

exec_id = 0
def test(args):
    global queue
    global exec_id
    print(args)
    #timeout, ((cur_test, total_tests,), (test_id, platform_id, knl, tlist, test_fn, max_flop_rate, device_latency, device_memory_bandwidth,),eval_str) = args
    #comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
    # From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD (this probably doesn't matter though. using COMM_WORLD should prevent
    # re-use of any GPU?
    
    eval_str = args["eval_str"]
    platform_id = args["platform_id"]
    #print("EVAL STR:", eval_str)

    if queue is None:
        # Maybe can just pass a function as an arg
        print("Queue is none. Initializing queue")
        # This is semi-broken because we aren't assured the device list is always
        # ordered the same, needs to order the devices by some pci_id first
        # Also, are pids contiguous?
        if eval_str == "mpi_comm_executor" or eval_str == "mpi_pool_executor":
            import mpi4py.MPI as MPI
            comm = MPI.COMM_WORLD
            exec_id = comm.Get_rank()
        elif eval_str == "charm4py_pool_executor":
            from charm4py import charm
            exec_id = charm.myPe()
        elif eval_str == "processpool":
            from os import getpid
            exec_id = getpid()
        elif eval_str == "threadpool":
            from threading import get_native_id
            exec_id = get_native_id()
        else:
            exec_id = 0

        set_queue(exec_id, platform_id)
        
        assert queue is not None


    print("EXECUTOR ID", exec_id)

    cur_test = args["cur_test"]
    total_tests = args["total_tests"]

    #print(f"\nExecuting test {cur_test} of {total_tests}\n")
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

        print(self.knl)
        print(tlist)

        test_id, result = test(args)

        print("ENDING TEST")
        #results = run_single_param_set_v2(queue, knl, tlist, max_flop_rate=max_flop_rate,
        #            device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency, timeout=timeout)
        
        # Would be helpful if could return all of the data instead of only avg_time 
        return result["data"]["avg_time"]




def offline_tuning(in_queue, knl, platform_id, input_space, program_id=None, max_flop_rate=np.inf, device_memory_bandwidth=np.inf,
                     device_latency=0, timeout=None, save_path=None):

    global exec_id

    if program_id is None:
        from utils import unique_program_id
        pid = unique_program_id(knl)
    else:
        pid = program_id

    if save_path is None:
        save_path = "./"

    print(input_space)
    output_space = Space([Real(0.0, inf, name="avg_time")])
    #eval_str = "mpi_comm_executor"
    #eval_str = "mpi_pool_executor"
    #eval_str = "charm4py_pool_executor"
    #eval_str = "threadpool"
    #eval_str = "processpool"
    eval_str = "ray"

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
    output_file_base = save_path + "/" +  pid 
    searcher = AMBS(problem=at_problem, evaluator=eval_str, output_file_base=output_file_base)
    searcher.main()

    print("======FINISHING SEARCHING========")

    # Write best result to hjson file
    import csv
    csv_file_str = output_file_base + ".csv"
    best_result = None

    if exec_id == 0:

        with open(csv_file_str) as csvfile:

            # The results in the csv file aren't directly transformation
            # parameters. They kio and iio need to be changed.
            row_list = list(csv.reader(csvfile))
            column_names = row_list[0]
            rows = list(row_list)[1:]
            rows.sort(key=lambda row: row[-2])
            #batch_size,iii,iio,ji,kii,kio,objective,elapsed_sec
            p = dict(zip(column_names, [int(item) for item in rows[0][:-2]]))
            
            params = (p["batch_size"],
                      p["kio"]*p["kii"],
                      p["kii"],
                      p["iio"]*p["iii"],
                      p["iii"],
                      p["ji"],)

            from generators import get_trans_list
            trans_list = get_trans_list(knl, params)


            """
            test_id = get_test_id(trans_list)
            args = frozendict({"timeout": timeout,
                    "cur_test": None,
                    "total_tests": None,
                    "test_id": test_id,
                    "platform_id": platform_id,
                    "knl": knl,
                    "tlist": trans_list,
                    "test_fn": generic_test,
                    "max_flop_rate": max_flop_rate,
                    "device_latency": device_latency,
                    "device_memory_bandwidth": device_memory_bandwidth,
                    "eval_str": eval_str
                    })
            """

            #test_id, tdict = test(args)

            # Re-run to obtain performance data and null-kernel latency
            tdict = run_single_param_set_v2(in_queue, knl, trans_list, generic_test,
                        max_flop_rate=max_flop_rate,
                        device_memory_bandwidth=device_memory_bandwidth,
                        device_latency=device_latency,
                        timeout=timeout)
            
            from utils import dump_hjson
            hjson_file_str = save_path + "/" + pid + ".hjson"
            dump_hjson(hjson_file_str, tdict)

    print("======RETURNING FROM SEARCH========")

    return True
