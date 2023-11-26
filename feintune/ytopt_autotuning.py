
from dataclasses import dataclass
from typing import Union, Optional
from feintune.generators import get_trans_list
from feintune.run_tests import generic_test, run_single_param_set_v2
from autotune import TuningProblem
from autotune.space import *
from skopt.space import Real
from ytopt.search.ambs import AMBS, LibEnsembleTuningProblem, LibEnsembleAMBS
from ytopt.search import util
logger = util.conf_logger(__name__)

from immutabledict import immutabledict
# from ytopt.search.async_search import AsyncSearch

import hjson
import numpy as np
import os
import loopy as lp
from os.path import exists
from feintune.utils import convert, load_hjson, dump_hjson
from hashlib import md5
from random import shuffle

# No guarantee the value written to the csv file is the same
# as this value.
#max_double = np.finfo('f').max

numeric_type = Union[np.number, float, int]

queue = None
exec_id = 0

def set_queue(exec_id, platform_name):
    import pyopencl as cl

    global queue
    if queue is not None:
        raise ValueError("queue is already set")

    if platform_name is None:
        platforms = cl.get_platforms()
    else:
        platforms = [platform for platform in cl.get_platforms() if platform.name == platform_name]

    #platforms = cl.get_platforms()
    #print(platforms)
    #print(platform_num)
    #print(exec_id)
    
    gpu_devices = []
    for platform in platforms:
        #print(platform)
        gpu_devices = platform.get_devices(
            device_type=cl.device_type.GPU)
        if len(gpu_devices) > 0:
            break

    if len(gpu_devices) == 0:
        raise OSError("No gpus detected")

    # Not sure if gpu_devices has a defined order, so sort it by bus id to prevent
    # oversubscription of a GPU.
    # Need to check if this works.
    """
    if "NVIDIA" in gpu_devices[0].vendor:
        gpu_devices = sorted(gpu_devices, key=lambda d: d.pci_bus_id_nv)
    elif "Advanced Micro Devices" in d.vendor:
        gpu_devices = sorted(gpu_devices, key=lambda d: d.topology_amd.bus)
    else:
        print("Unrecognized vendor, not sorting GPU list")
    """

    ctx = cl.Context(devices=[gpu_devices[exec_id % len(gpu_devices)]])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


def get_test_id(tlist):
    return md5(str(tlist).encode()).hexdigest()


def test(args):

    if True:
        global queue
        global exec_id
        print(args)
        # timeout, ((cur_test, total_tests,), (test_id, platform_id, knl, tlist, test_fn, max_flop_rate, device_latency, device_memory_bandwidth,),eval_str) = args
        # comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
        # From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD (this probably doesn't matter though. using COMM_WORLD should prevent
        # re-use of any GPU?

        eval_str = args["eval_str"]
        platform_id = args["platform_id"]
        # print("EVAL STR:", eval_str)

        if queue is None:
            # Maybe can just pass a function as an arg
            print("Queue is none. Initializing queue")
            # This is semi-broken because we aren't assured the device list is always
            # ordered the same, needs to order the devices by some pci_id first
            # Also, are pids contiguous?
            if args["exec_id"] is None:
                if eval_str in {"mpi_comm_executor", "mpi_pool_executor", "mpi_libensemble", "mpi_libensemble_subprocess"}:
                        import mpi4py
                        # Prevent mpi4py from automatically initializing.
                        mpi4py.rc.initialize = False
                        import mpi4py.MPI as MPI
                        if not MPI.Is_initialized():
                            MPI.Init()

                        comm = MPI.COMM_WORLD
                        exec_id = comm.Get_rank()
                elif eval_str == "charm4py_pool_executor":
                    from charm4py import charm
                    exec_id = charm.myPe()
                elif eval_str == "processpool" or eval_str == "subprocess":
                    from os import getpid
                    exec_id = getpid()
                elif eval_str == "threadpool":
                    from threading import get_native_id
                    exec_id = get_native_id()
                else:
                    exec_id = 0
            else:
                exec_id = args["exec_id"]

            set_queue(exec_id, platform_id)

            assert queue is not None

        print("EXECUTOR ID", exec_id)

        cur_test = args["cur_test"]
        total_tests = args["total_tests"]

        #import sys
        #sys.stderr.write(str(queue))
        #sys.stderr.write(str(queue.device))

        #result =  {"data": {"avg_time_predicted": 1.0}}
        # print(f"\nExecuting test {cur_test} of {total_tests}\n")
        #"""
        result = run_single_param_set_v2(queue, args["knl"], args["tlist"], args["test_fn"],
                                     max_flop_rate=args["max_flop_rate"],
                                     device_memory_bandwidth=args["device_memory_bandwidth"],
                                     device_latency=args["device_latency"],
                                     timeout=args["timeout"],
                                     method=args["method"],#"subprocess",#"thread",  # "subprocess",#None
                                     run_single_batch=False,
                                     error_return_time=args["error_return_time"],
                                     # Speed things up a bit by not measuring the latency. Only works
                                     # for method=None for now.
                                     measure_latency=False if args["method"] is None else True)
        #"""
    else:
        result = {"data": {"avg_time_predicted": 0.0}}

    return args["test_id"], result

# For libensemble to be robust, this needs to be called by an executor I think.
@dataclass
class ObjectiveFunction(object):

    knl: object
    eval_str: Optional[str] = None
    platform_id: Optional[str] = None
    max_flop_rate: numeric_type = np.inf
    device_memory_bandwidth: numeric_type = np.inf
    device_latency: numeric_type = 0
    timeout: Optional[numeric_type] = None
    error_return_time: Optional[numeric_type] = 999
    environment_failure_flag: Optional[numeric_type] = 998
    exec_id: Optional[int] = None
    method: Optional[str] = None

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

        tlist = get_trans_list(self.knl, params, prefetch=p["prefetch"], group_idof=p["group_idofs"],
                                iel_ilp=p["iel_ilp"], idof_ilp=p["idof_ilp"], swap_local=p["swap_local"])

        test_id = get_test_id(tlist)

        print("BEGINNING TEST")
        # args = (self.timeout, ((None, None,), (test_id, self.platform_id, self.knl, tlist,
        #        generic_test, self.max_flop_rate, self.device_latency, self.device_memory_bandwidth,),),self.eval_str,)
        args = immutabledict({
                           "timeout": self.timeout,
                           "error_return_time": self.error_return_time,
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
                           "eval_str": self.eval_str,
                           "method": self.method,
                           "exec_id": self.exec_id,
                           })

        # print(self.knl)
        print(tlist)

        if True:
            try:
                test_id, result = test(args)
            except OSError as e:
                print(e)
                return self.environment_failure_flag

            print("ENDING TEST")
            #if result["data"]["avg_time_predicted"] > self.timeout:
            #    exit()

            # Would be helpful if could return all of the data instead of only avg_time
            return result["data"]["avg_time_predicted"]
        else:
            return 1


def ytopt_tuning(in_queue, knl, platform_id, input_space, program_id=None, normalized_program_id=None, max_flop_rate=np.inf, device_memory_bandwidth=np.inf, device_latency=0, timeout=None, save_path=None, max_evals=100, required_new_evals=None, eval_str="threadpool"):

    if required_new_evals is None:
        required_new_evals = max_evals

    import mpi4py
    # Prevent mpi4py from automatically initializing.
    mpi4py.rc.initialize = False
    import mpi4py.MPI as MPI
    comm = None

    global exec_id

    from feintune.utils import unique_program_id
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
    # eval_str = "mpi_comm_executor"
    # eval_str = "mpi_pool_executor"
    # eval_str = "charm4py_pool_executor"
    #eval_str = "threadpool"
    # eval_str = "processpool"
    # eval_str = "ray"

    # Note that using Popen (forking) with MPI often results in strange errors and unpredictable crashes. 
    # Only use subprocess with non-MPI executions
    import feintune
    dirname = os.path.dirname(feintune.__file__)
    if eval_str == "local_libensemble":
        #wrapper_script = str(os.path.join(dirname, "run_objective_fn_sh_mem.py"))
        wrapper_script = str(os.path.join(dirname, "run_objective_fn_disk.py"))
        method = None
    elif eval_str == "mpi_libensemble_subprocess":
        # This is not guaranteed to work as forking within an MPI process has undefined behavior
        # Spectrum MPI (Open MPI based) on Lassen seems to work but MPICH does not tolerate it well.
        # For SS11
        # export CXI_FORK_SAFE=1
        # export CXI_FORK_SAFE_HP=1
        # and for SS10
        # export IBV_FORK_SAFE=1
        # export RDMAV_HUGEPAGES_SAFE=1
        # may or may not help to address this problem.
        # (On Crusher, it eliminates the segfaults but there are still MPICH errors
        # See https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#known-issues
        #wrapper_script = str(os.path.join(dirname, "run_objective_fn_disk.py"))
        wrapper_script = str(os.path.join(dirname, "run_objective_fn_sh_mem.py"))
        method = None
    else:
        wrapper_script = None
        method = "thread"#"subprocess"

    environment_failure_flag = 998
    error_return_time = 999#np.inf is timeout is None else timeout+1
    obj_func = ObjectiveFunction(knl, eval_str=eval_str, platform_id=platform_id, max_flop_rate=max_flop_rate,
                                 device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency,
                                 timeout=timeout, method=method, error_return_time=error_return_time, environment_failure_flag=environment_failure_flag)

    # If want to time step (advance simulation) while tuning, need to use the actual kernel and data, so each node is
    # independent... but maybe the results can be unified after kernel execution? Each rank should have a different
    # seed then. But if the tuning is done in a different process then the results won't be available anyway, at
    # least not without piping it to the main process or saving it to a file, which would require saving it off the
    # the GPU or somehow sharing the buffer between processes (which is impossible).

    # Could have the arg sizes be a parameter and then fix the sizes for the kernel that must be run
    # (slightly different config space for each rank)

    """
    at_problem = TuningProblem(
        task_space=None,
        input_space=input_space,
        output_space=output_space,
        objective=obj_func,
        constraints=None,
        model=None)
    """

    #"""
    at_problem = LibEnsembleTuningProblem(
        task_space=None,
        input_space=input_space,
        output_space=output_space,
        objective=obj_func,
        constraints=None,
        model=None,
        wrapper_script=wrapper_script)

    #"""

    # for entry in range(10):
    #    p = input_space.sample_configuration(1)
    #    print(p)
    #    print(obj_func(p))

    output_file_base = save_path + "/" + npid
    # learner = "DUMMY"
    # learner = "RF"
    # learner = "ET"
    learner = "GBRT"
    # learner = "GP" # Doesn't work with ConfigSpace
    seed = 12345

    import csv
    csv_file_str = output_file_base + ".csv"

    initial_observations = []

    # from .generators import get_inames
    # nelem = i

    # Maybe use pandas instead?
    if exists(csv_file_str):
        #from feintune.utils import mpi_read_all

        #csvfile = mpi_read_all(csv_file_str)
        print(f"Loading saved data from {csv_file_str}")
        with open(csv_file_str) as csvfile:
            row_list = list(csv.reader(csvfile))
            column_names = row_list[0]
            # assert column_names[-4] == "num_elements"
            for row in row_list[1:]:
                p = dict(zip(column_names, [int(item) for item in row[:-2]]))
                #if float(row[-2]) <= timeout:  # Eliminate
                initial_observations.append((p, float(row[-2]),))
                # if int(row[-4]) == nelem:
                #    pre_existing_evals += 1

        num_random = 0
    else:
        print("No saved data found.")
        num_random = None

    pre_existing_evals = len(initial_observations)
    max_evals = min(max_evals, pre_existing_evals + required_new_evals)

    # Note that the initial observations count toward max_evals.
    libE_specs = {"disable_log_files": True,
                  "save_H_and_persis_on_abort": False,
                  #"nworkers": 2
                  #"comms": "local"
                 }

    if eval_str == "mpi_libensemble" or eval_str == "mpi_libensemble_subprocess":
        if not MPI.Is_initialized():
            MPI.Init()

        comm = MPI.COMM_WORLD

        assert comm.Get_size() >= 3
        if num_random is None:
            num_random = min(2, 2*(comm.Get_size() - 2))
        searcher = LibEnsembleAMBS(problem=at_problem, output_file_base=output_file_base, learner=learner,
                    set_seed=seed, max_evals=max_evals + num_random, set_NI=num_random, initial_observations=initial_observations,
                    error_flag_val=error_return_time, environment_failure_flag=environment_failure_flag,
                    libE_specs=libE_specs)
    elif eval_str == "local_libensemble":

        from libensemble.resources.env_resources import EnvResources
        #from libensemble.resources.resources import GlobalResources
        #nodelist = GlobalResources.get_global_nodelist()

        from libensemble.resources.node_resources import get_sub_node_resources
        envR = EnvResources()
        nodelist = envR.get_nodelist()
        #print(envR.get_nodelist())

        print(nodelist)

        nnodes = max(1, len(nodelist))

        sn_resources = get_sub_node_resources() # Assume all nodes are identical
        if len(sn_resources) == 3:
            gpus_per_node = sn_resources[-1]
        else:
            # Although on rocinante it should return 1 rather than 0
            gpus_per_node = 1

        assert gpus_per_node > 0

        # If the number of available resource sets exceeds half of the
        # available threads, then switch to dedicated mode.
        num_resource_sets = gpus_per_node*nnodes
        if (num_resource_sets + 1) > sn_resources[1] // 2:
            libE_specs["dedicated_mode"]=True
            num_resource_sets -= gpus_per_node

        overtasking_factor = 1
        # Limit the numbers of workers per process (sn_resources[0]) or thread (sn_resources[1])
        nworkers = min(num_resource_sets + 1, overtasking_factor*sn_resources[1] - 1) # 1 Manager (not a worker), 1 worker for persistent generator, more workers for the gpus
        n_sim_workers = nworkers - 1
        if num_random is None:
            num_random = 2*n_sim_workers

        print(f"Running with {nworkers} workers.")
        searcher = LibEnsembleAMBS(problem=at_problem, output_file_base=output_file_base, learner=learner,
                    set_seed=seed, max_evals=max_evals + num_random, set_NI=num_random, initial_observations=initial_observations,
                    error_flag_val=error_return_time, environment_failure_flag=environment_failure_flag,
                    libE_specs=libE_specs | {"nworkers": nworkers, "comms": "local", "num_resource_sets": num_resource_sets})
    else:
        if num_random is None:
            num_random = 2
        searcher = AMBS(problem=at_problem, evaluator=eval_str, output_file_base=output_file_base, learner=learner, error_flag_val=error_return_time,
                    environment_failure_flag=environment_failure_flag,
                    set_seed=seed, max_evals=max_evals, set_NI=num_random, initial_observations=initial_observations)


    update_hjson = True

    if pre_existing_evals < max_evals:
        print("==========BEGINNING SEARCH=============")
        searcher.main()
        print("======FINISHING SEARCHING========")
    else:
        print("=======SKIPPING SEARCH: EXISTING EVALS >= MAX_EVALS")
        print(pre_existing_evals, max_evals)

        hjson_file_str = save_path + "/" + pid + "_full" + ".hjson"
        if exists(hjson_file_str):
            from feintune.utils import load_hjson
            current_hjson = load_hjson(hjson_file_str)
            cur_data = current_hjson["data"]
            if "frac_roofline_flop_rate" in cur_data:
                print("UPDATED HJSON FILE ALREADY EXISTS, SKIPPING GENERATION")
                update_hjson = False

    if comm is not None:
        comm.Barrier()

    # Not sure if this works for ray
    if update_hjson and (comm is not None and comm.Get_rank() == 0 and "mpi" in eval_str) or "mpi" not in eval_str:

        # Write best result to hjson file
        with open(csv_file_str) as csvfile:

            # The results in the csv file aren't directly transformation
            # parameters. The kio and iio need to be changed.
            row_list = list(csv.reader(csvfile))
            column_names = row_list[0]
            # rows = [row for row in list(row_list)[1:] if int(row[-4]) == nelem]
            rows = list(row_list)[1:]
            rows.sort(key=lambda row: row[-2])

            if (timeout is None) or (float(rows[0][-2]) < timeout):
                # batch_size,iii,iio,ji,kii,kio,objective,elapsed_sec
                p = dict(zip(column_names, [int(item)
                         for item in rows[0][:-2]]))
                #p = dict(zip(column_names, rows[0]))

                params = (p["batch_size"],
                          p["kio"]*p["kii"],
                          p["kii"],
                          p["iio"]*p["iii"],
                          p["iii"],
                          p["ji"],)

                trans_list = get_trans_list(knl, params, prefetch=p["prefetch"], group_idof=p["group_idofs"],
                                iel_ilp=p["iel_ilp"], idof_ilp=p["idof_ilp"], swap_local=p["swap_local"])

                """
                test_id = get_test_id(trans_list)
                args = immutabledict({"timeout": timeout,
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

                # test_id, tdict = test(args)

                # Re-run to obtain performance data and null-kernel latency
                # since ytopt doesn't appear to have a way to return ancillary data.
                # Could have each rank/process/thread write to a file and then recombine the
                # results. Note, this writes to pid.hjson, not npid.hjson
                #hjson_file_str = save_path + "/" + pid + ".hjson"
                # Kernels that use too much memory still aren't prevented from running.
                # In particular, if the timout time is None or infinity
                #update_hjson = True#prexisting_evals < max_evals#True
                """ # Broken currently.
                if exists(hjson_file_str):
                    from feintune.utils import load_hjson
                    current_hjson = load_hjson(hjson_file_str)
                    current_transformations = current_hjson["transformations"]
                    print("CURRENT TRANSFORMATIONS")
                    print(current_transformations)
                    print("NEW TRANSFORMATIONS")
                    print(trans_list)
                    if trans_list == current_transformations:
                        update_hjson = False
                        print("Setting hjson update to false")
                        # exit()
                """

                #if update_hjson:

                hjson_file_str = save_path + "/" + pid + ".hjson"
                tdict = run_single_param_set_v2(in_queue, knl, trans_list, generic_test,
                                                max_flop_rate=max_flop_rate,
                                                device_memory_bandwidth=device_memory_bandwidth,
                                                device_latency=device_latency,
                                                timeout=timeout,
                                                method=None,#"thread",  # "subprocess",
                                                run_single_batch=True,
                                                error_return_time=timeout + 1)
                if tdict["data"]["avg_time_predicted"] < timeout:
                    from feintune.utils import dump_hjson
                    dump_hjson(hjson_file_str, tdict)

                    # If the single batch kernel didn't time out
                    # run the full kernel with those transformations.
                    if True:  # See what the performance is with the full kernel.
                        # not exists(hjson_file_str) or pre_existing_evals < max_evals:
                        if True:

                            hjson_file_str = save_path + "/" + pid + "_full" + ".hjson"
                            print("GENERATING AND EXECUTING FULL KERNEL")
                            tdict = run_single_param_set_v2(in_queue, knl, trans_list, generic_test,
                                                            max_flop_rate=max_flop_rate,
                                                            device_memory_bandwidth=device_memory_bandwidth,
                                                            device_latency=device_latency,
                                                            timeout=None,
                                                            method=None,#"thread",  # "subprocess",
                                                            run_single_batch=False,
                                                            error_return_time=timeout + 1)
                            print("DONE GENERATING AND EXECUTING FULL KERNEL")
                            if (timeout is None) or (tdict["data"]["avg_time_predicted"] < timeout):
                                dump_hjson(hjson_file_str, tdict)
                            else:
                                print(
                                    "Run return error return time. Not dumping to hjson.")

                else:
                    print("Run return error return time. Not dumping to hjson.")


                hjson_file_str = save_path + "/" + pid + "_default" + ".hjson"
                if not exists(hjson_file_str):
                    from meshmode.array_context import PrefusedFusionContractorArrayContext
                    actx = PrefusedFusionContractorArrayContext(in_queue)
                    knl_with_default_transformations = actx.transform_loopy_program(
                        knl)

                    print("GENERATING AND EXECUTING DEFAULT TRANSFORMED KERNEL")
                    tdict = run_single_param_set_v2(in_queue, knl_with_default_transformations, [], generic_test,
                                                    max_flop_rate=max_flop_rate,
                                                    device_memory_bandwidth=device_memory_bandwidth,
                                                    device_latency=device_latency,
                                                    timeout=None,
                                                    method=None,#"thread",  # "subprocess",
                                                    run_single_batch=False,
                                                    error_return_time=timeout + 1)
                    print("DONE GENERATING AND EXECUTING DEFAULT TRANSFORMED KERNEL")

                    if tdict["data"]["avg_time_predicted"] < timeout:
                        from feintune.utils import dump_hjson
                        dump_hjson(hjson_file_str, tdict)
                    else:
                        print("Run return error return time. Not dumping to hjson.")

    #if "mpi" in eval_str:
    #    print("WAITING AT BARRIER")
        #comm = MPI.COMM_WORLD
    if comm is not None:
        comm.Barrier()
    print("======RETURNING FROM SEARCH========")
    # exit()
    return True
