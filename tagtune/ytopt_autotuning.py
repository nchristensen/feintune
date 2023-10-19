import mpi4py.MPI as MPI
from dataclasses import dataclass
from typing import Union, Optional
from tagtune.generators import get_trans_list
from tagtune.run_tests import generic_test, run_single_param_set_v2
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
from tagtune.utils import convert, load_hjson, dump_hjson
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
            timeout=args["timeout"],
            method="thread",#"subprocess",#None
            run_single_batch=True,
            error_return_time=args["timeout"])

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
        # num_elements is only used for training the model,
        # not for running the tests
        params = (p["batch_size"],
                  p["kio"]*p["kii"],
                  p["kii"],
                  p["iio"]*p["iii"],
                  p["iii"],
                  p["ji"],)
        tlist = get_trans_list(self.knl, params, prefetch=p["prefetch"])
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

        #print(self.knl)
        print(tlist)

        test_id, result = test(args)

        print("ENDING TEST")
        #results = run_single_param_set_v2(queue, knl, tlist, max_flop_rate=max_flop_rate,
        #            device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency, timeout=timeout)
        
        # Would be helpful if could return all of the data instead of only avg_time 
        return result["data"]["avg_time_predicted"]



# TODO: Change default max_evals
def ytopt_tuning(in_queue, knl, platform_id, input_space, program_id=None, normalized_program_id=None, max_flop_rate=np.inf, device_memory_bandwidth=np.inf, device_latency=0, timeout=None, save_path=None, max_evals=10, required_new_evals=0, eval_str="threadpool"):

    global exec_id

    from tagtune.utils import unique_program_id
    if program_id is None:
        pid = unique_program_id(knl, attempt_normalization=False)
    else:
        pid = program_id

    if normalized_program_id is None:
        npid = unique_program_id(knl, attempt_normalization=True)
    else:
        npid = normalized_program_id


    if save_path is None:
        save_path = "./"

    print(input_space)
    output_space = Space([Real(0.0, inf, name="avg_time_predicted")])
    #eval_str = "mpi_comm_executor"
    #eval_str = "mpi_pool_executor"
    #eval_str = "charm4py_pool_executor"
    #eval_str = "threadpool"
    #eval_str = "processpool"
    #eval_str = "ray"

    obj_func = ObjectiveFunction(knl, eval_str=eval_str, platform_id=platform_id, max_flop_rate=max_flop_rate,
                                    device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency,
                                    timeout=timeout)

    # If want to time step (advance simulation) while tuning, need to use the actual kernel and data, so each node is
    # independent... but maybe the results can be unified after kernel execution? Each rank should have a different
    # seed then. But if the tuning is done in a different process then the results won't be available anyway, at
    # least not without piping it to the main process or saving it to a file, which would require saving it off the
    # the GPU or somehow sharing the buffer between processes (which is impossible).

    # Could have the arg sizes be a parameter and then fix the sizes for the kernel that must be run
    # (slightly different config space for each rank)

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
     
    output_file_base = save_path + "/" +  npid 
    #learner = "DUMMY"
    #learner = "RF"
    #learner = "ET"
    learner = "GBRT"
    #learner = "GP" # Doesn't work with ConfigSpace
    seed = 12345

    import csv
    csv_file_str = output_file_base + ".csv"

    initial_observations = []

    #from .generators import get_inames
    #nelem = i

    # Maybe use pandas instead?
    pre_existing_evals = 0
    if exists(csv_file_str):
        print(f"Loading saved data from {csv_file_str}")
        with open(csv_file_str) as csvfile:
            row_list = list(csv.reader(csvfile))
            column_names = row_list[0]
            #assert column_names[-4] == "num_elements"
            for row in row_list[1:]:
                p = dict(zip(column_names, [int(item) for item in row[:-2]]))
                if float(row[-2]) != timeout: # Eliminate
                    initial_observations.append((p, float(row[-2]),))
                #if int(row[-4]) == nelem:
                #    pre_existing_evals += 1

        num_random = 0
    else:
        print("No saved data found.")
        num_random = 2

    max_evals = min(max_evals, pre_existing_evals + required_new_evals)

    # Note that the initial observations count toward max_evals.
    # --Is this actually true?
    searcher = AMBS(problem=at_problem, evaluator=eval_str, output_file_base=output_file_base, learner=learner, set_seed=seed, max_evals=max_evals, set_NI=num_random, initial_observations=initial_observations)

    if pre_existing_evals < max_evals:
        print("==========BEGINNING SEARCH=============")
        searcher.main()
        print("======FINISHING SEARCHING========")
    else:
        print("=======SKIPPING SEARCH: EXISTING EVALS >= MAX_EVALS")
        print(pre_existing_evals, max_evals)

    best_result = None

    # Not sure if this works for ray
    if (exec_id == 0 and "mpi" in eval_str) or "mpi" not in eval_str:

        # Write best result to hjson file
        with open(csv_file_str) as csvfile:

            # The results in the csv file aren't directly transformation
            # parameters. The kio and iio need to be changed.
            row_list = list(csv.reader(csvfile))
            column_names = row_list[0]
            #rows = [row for row in list(row_list)[1:] if int(row[-4]) == nelem]
            rows = list(row_list)[1:]
            rows.sort(key=lambda row: row[-2])

            if (timeout is None) or (float(rows[0][-2]) < timeout):
                #batch_size,iii,iio,ji,kii,kio,objective,elapsed_sec
                p = dict(zip(column_names, [int(item) for item in rows[0][:-2]]))
                
                params = (p["batch_size"],
                          p["kio"]*p["kii"],
                          p["kii"],
                          p["iio"]*p["iii"],
                          p["iii"],
                          p["ji"],)

                trans_list = get_trans_list(knl, params, prefetch=p["prefetch"])


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
                # since ytopt doesn't appear to have a way to return ancillary data.
                # Could have each rank/process/thread write to a file and then recombine the 
                # results. Note, this writes to pid.hjson, not npid.hjson
                hjson_file_str = save_path + "/" + pid + ".hjson"
                # Kernels that use too much memory still aren't prevented from running.
                # In particular, if the timout time is None or infinity

                if True:#not exists(hjson_file_str) or pre_existing_evals < max_evals:
                    tdict = run_single_param_set_v2(in_queue, knl, trans_list, generic_test,
                                max_flop_rate=max_flop_rate,
                                device_memory_bandwidth=device_memory_bandwidth,
                                device_latency=device_latency,
                                timeout=timeout,
                                method="thread",#"subprocess",
                                run_single_batch=True,
                                error_return_time=timeout)
                    if tdict["data"]["avg_time_predicted"] < timeout:
                        from tagtune.utils import dump_hjson
                        dump_hjson(hjson_file_str, tdict)
                    else:
                        print("Run return error return time. Not dumping to hjson.")

                if True: # See what the performance is with the full kernel.
                    hjson_file_str = save_path + "/" + pid + "_full" + ".hjson"
                    if True:#not exists(hjson_file_str) or pre_existing_evals < max_evals:
            
                        print("GENERATING AND EXECUTING FULL KERNEL")
                        tdict = run_single_param_set_v2(in_queue, knl, trans_list, generic_test,
                                    max_flop_rate=max_flop_rate,
                                    device_memory_bandwidth=device_memory_bandwidth,
                                    device_latency=device_latency,
                                    timeout=timeout,
                                    method="thread",#"subprocess",
                                    run_single_batch=False,
                                    error_return_time=timeout)
                        print("DONE GENERATING AND EXECUTING FULL KERNEL")
                        if tdict["data"]["avg_time_predicted"] < timeout:
                            from tagtune.utils import dump_hjson
                            dump_hjson(hjson_file_str, tdict)
                        else:
                            print("Run return error return time. Not dumping to hjson.")
                if True:
                    from meshmode.array_context import PrefusedFusionContractorArrayContext
                    hjson_file_str = save_path + "/" + pid + "_default" + ".hjson"
                    actx = PrefusedFusionContractorArrayContext(in_queue)
                    knl_with_default_transformations = actx.transform_loopy_program(knl)

                    print("GENERATING AND EXECUTING DEFAULT TRANSFORMED KERNEL")
                    tdict = run_single_param_set_v2(in_queue, knl_with_default_transformations, [], generic_test,
                                max_flop_rate=max_flop_rate,
                                device_memory_bandwidth=device_memory_bandwidth,
                                device_latency=device_latency,
                                timeout=timeout,
                                method="thread",#"subprocess",
                                run_single_batch=False,
                                error_return_time=timeout)
                    print("DONE GENERATING AND EXECUTING DEFAULT TRANSFORMED KERNEL")

                    if tdict["data"]["avg_time_predicted"] < timeout:
                        from tagtune.utils import dump_hjson
                        dump_hjson(hjson_file_str, tdict)
                    else:
                        print("Run return error return time. Not dumping to hjson.")

    if "mpi" in eval_str:
        print("WAITING AT BARRIER") 
        comm = MPI.COMM_WORLD
        comm.Barrier()                
    print("======RETURNING FROM SEARCH========")
    #exit()
    return True
