from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
from charm4py.pool import PoolScheduler, Pool
from charm4py.charm import Charm, CharmRemote
#from charm4py.chare import GROUP, MAINCHARE, ARRAY, CHARM_TYPES, Mainchare, Group, ArrayMap
#from charm4py.sections import SectionManager
#import inspect
#import sys
import hjson
import pyopencl as cl
import numpy as np
#import grudge.loopy_dg_kernels as dgk
import os
#import grudge.grudge_array_context as gac
import loopy as lp
from os.path import exists
from run_tests import run_single_param_set_v2, generic_test
from utils import convert, load_hjson, dump_hjson
from hashlib import md5
from random import shuffle
#from grudge.execution import diff_prg, elwise_linear

# Makes one PE inactive on each host so the number of workers is the same on all hosts as
# opposed to the basic PoolScheduler which has one fewer worker on the host with PE 0.
# This can be useful for running tasks on a GPU cluster for example.
class BalancedPoolScheduler(PoolScheduler):

    def __init__(self):
       super().__init__()
       n_pes = charm.numPes()
       n_hosts = charm.numHosts()
       pes_per_host = n_pes // n_hosts

       assert n_pes % n_hosts == 0 # Enforce constant number of pes per host
       assert pes_per_host > 1 # We're letting one pe on each host be unused

       self.idle_workers = set([i for i in range(n_pes) if not i % pes_per_host == 0 ])
       self.num_workers = len(self.idle_workers)

# Use all PEs including PE 0 
class AllPEsPoolScheduler(PoolScheduler):

    def __init__(self):
       super().__init__()
       n_pes = charm.numPes()
       n_hosts = charm.numHosts()

       self.idle_workers = set(range(n_pes))
       self.num_workers = len(self.idle_workers)

queue = None
def set_queue(pe_num, platform_num):
    global queue
    if queue is not None:
        raise ValueError("queue is already set")
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    #queue = 
    #return queue

# Breaks on Lassen
#import mpi4py.MPI as MPI
#comm = MPI.COMM_WORLD
#queue = get_queue(comm.Get_rank(), 0)
#queue = get_queue(0,0)

"""
def do_work(args):
    params = args[0]
    knl = args[1]
    queue = get_queue(charm.myPe())
    print("PE: ", charm.myPe())
    avg_time, transform_list = dgk.run_tests.apply_transformations_and_run_test(queue, knl, dgk.run_tests.generic_test, params)
    return avg_time, params
"""

def get_test_id(tlist):
    return md5(str(tlist).encode()).hexdigest()

def test(args):
    print(args)
    timeout, ((cur_test, total_tests), (test_id, platform_id, knl, tlist, test_fn, max_flop_rate, device_latency, device_memory_bandwidth,),) = args
    #comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
    # From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD
    global queue
    if queue is None:
        print("Queue is none. Initializing queue")
        set_queue(charm.myPe(), platform_id)
        assert queue is not None
    else:
        print("Using prexisting queue")
    
    print(f"\nExecuting test {cur_test} of {total_tests}\n")
    result = run_single_param_set_v2(queue, knl, tlist, test_fn,
            max_flop_rate=max_flop_rate, 
            device_memory_bandwidth=device_memory_bandwidth,
            device_latency=device_latency,
            timeout=timeout)
    #print(mem_top())
    #h = hpy()
    #print(h.heap())
    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)
    #del knl
    #del args

    #result = [10,10,10]
    #test_id = get_test_id(tlist)
    return test_id, result

#def test(args):
#    platform_id, knl, tlist, test_fn = args
#    #queue = get_queue(charm.myPe(), platform_id)
#    result = run_single_param_set_v2(queue, knl, tlist, test_fn) 
#    return result

def unpickle_kernel(fname):
    from pickle import load
    f = open(fname, "rb")
    program = load(f)
    f.close()
    return program

def autotune_pickled_kernels(path, platform_id, actx_class, comm):
    from os import listdir
    dir_list = listdir(path)
    for f in dir_list:
        if f.endswith(".pickle"):
            fname = path + "/" + f
            print("===============================================")
            print("Autotuning", fname)
            knl = unpickle_kernel(fname)
            knl_id = f.split(".")[0]
            knl_id = knl_id.split("_")[-1]
            print("Kernel ID", knl_id)
            print("New kernel ID", gac.unique_program_id(knl))
            
            assert knl_id == gac.unique_program_id(knl)
            knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
            knl = gac.set_memory_layout(knl)
            assert knl_id == gac.unique_program_id(knl)

            print(knl)
            pid = gac.unique_program_id(knl)
            hjson_file_str = f"hjson/{knl.default_entrypoint.name}_{pid}.hjson"
            if not exists(hjson_file_str):
                parallel_autotune(knl, platform_id, actx_class, comm)
            else:
                print("hjson file exists, skipping")

def parallel_autotune(knl, platform_id, trans_list_list, program_id=None, max_flop_rate=None, device_latency=None, device_memory_bandwidth=None, save_path=None, timeout=30):

    initial_timeout = timeout

    if save_path is None:
        save_path = "./hjson"

    # Create queue, assume all GPUs on the machine are the same
    #platforms = cl.get_platforms()
    #gpu_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    #n_gpus = len(gpu_devices)
    #ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
    #profiling = cl.command_queue_properties.PROFILING_ENABLE
    #queue = cl.CommandQueue(ctx, properties=profiling)    

    """
    import pyopencl.tools as cl_tools
    actx = actx_class(
        comm,
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    #knl = gac.fix_program_parameters(knl)
    #knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
    knl = gac.set_memory_layout(knl)
    pid = gac.unique_program_id(knl)
    os.makedirs(os.getcwd() + "/hjson", exist_ok=True)
    hjson_file_str = f"hjson/{knl.default_entrypoint.name}_{pid}.hjson"
    """

    assert charm.numPes() > 1
    assert knl.default_entrypoint.options.no_numpy
    assert knl.default_entrypoint.options.return_dict

    #assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    #assert charm.numPes() <= charm.numHosts()*(len(gpu_devices) + 1)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes

    #from run_tests import run_single_param_set
    
    #tlist_generator, pspace_generator = actx.get_generators(knl)
    #params_list = pspace_generator(actx.queue, knl)
    if program_id is None:
        from utils import unique_program_id
        pid = unique_program_id(knl)
    else:
        pid = program_id
    #knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
    #knl = gac.set_memory_layout(knl)
    #os.makedirs(os.getcwd() + "/hjson", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    hjson_file_str = f"{save_path}/{pid}.hjson"
    test_results_file =f"{save_path}/{pid}_tests.hjson"

    print("Final result file:", hjson_file_str)
    print("Test results file:", test_results_file)

    results_dict = load_hjson(test_results_file) if exists(test_results_file) else {}

    #for tlist in trans_list_list:
    #    print(get_test_id(tlist) in results_dict)
    #print(results_dict.keys())
    #exit()

    args = [(get_test_id(tlist), platform_id, knl, tlist, generic_test, max_flop_rate, device_latency, device_memory_bandwidth,) for tlist in trans_list_list if get_test_id(tlist) not in results_dict]
    results = list(results_dict.items())

    ntransforms = len(args)

    shuffle(args)
    #args = list(reversed(args))

    # Number so the tasks can display how many tasks remain
    args = [((ind + 1, ntransforms,), arg,) for ind, arg in enumerate(args)]

    sort_key = lambda entry: entry[1]["data"]["avg_time"]
    segment_size = 5*(charm.numPes() - 1) # Arbitrary.

    # We should be able to unify the mpi4py and charm4py versions. The only 
    # difference is how the pool is created
    pool_proxy = Chare(PoolScheduler, onPE=0)
    mypool = Pool(pool_proxy)

    if len(trans_list_list) > 0: # Guard against empty list
        start_ind = 0
        end_ind = min(segment_size, len(args))
        while start_ind < end_ind:
            # Get new result segment
            args_segment = args[start_ind:end_ind]
            args_segment_with_timeout = [(timeout, args,) for args in args_segment]
            partial_results = list(mypool.map(test, args_segment_with_timeout, chunksize=1))
            results = results + partial_results
            results.sort(key=sort_key)
            
            #timeout = min(timeout, 10*results[0][1]["data"]["wall_clock_time"])

            timeout = min(initial_timeout, 3*results[0][1]["data"]["wall_clock_time"])

            # Add to existing results
            start_ind += segment_size
            end_ind = min(end_ind + segment_size, len(args))

            # Write the partial results to a file
            dump_hjson(test_results_file, dict(results)) 

    results.sort(key=sort_key)
    result = results[0][1] if len(results) > 0 else {"transformations": {}, "data": {}}
    # Write the final result to a file 
    print(result)
    dump_hjson(hjson_file_str, result)

    return result

"""
        results = list(mypool.map(test, args, chunksize=1))
        results.sort(key=sort_key)

        # Workaround for pocl CUDA bug
        # whereby times are imprecise
        # Fixed in newer version of pocl
        ret_index = 0
        for i, result in enumerate(results):
            if result["data"]["avg_time"] > 1e-7:
                ret_index = i
                break

        result = results[ret_index]

    #od = {"transformations": transformations, "data": data}
    if True:#comm.Get_rank() == 0:
        print(result)
        out_file = open(hjson_file_str, "wt+")
        hjson.dump(result, out_file, default=convert)
        out_file.close()

    return result



    pool_proxy = Chare(PoolScheduler, onPE=0)
    mypool = Pool(pool_proxy)
    if len(args) > 0: # Guard against empty list
        results = mypool.map(test, args[:5])

        sort_key = lambda entry: entry[0]
        results.sort(key=sort_key)
        
        #for r in results:
        #    print(r)
        # Workaround for pocl CUDA bug
        # whereby times are imprecise
        ret_index = 0
        for i, result in enumerate(results):
            if result[0] > 1e-7:
                ret_index = i
                break

        avg_time, transformations, data = results[ret_index]
    else:
        transformations = {}
    
    #od = {"transformations": transformations}
    #out_file = open(hjson_file_str, "wt+")
    #hjson.dump(od, out_file,default=convert)
    #out_file.close()

    return transformations
"""
"""
def main(args):

    # Create queue, assume all GPUs on the machine are the same
    platforms = cl.get_platforms()
    platform_id = 0
    gpu_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    n_gpus = len(gpu_devices)
    ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
    profiling = cl.command_queue_properties.PROFILING_ENABLE
    queue = cl.CommandQueue(ctx, properties=profiling)    
   
    assert charm.numPes() > 1
    #assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    assert charm.numPes() <= charm.numHosts()*(len(gpu_devices) + 1)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes
    
    from grudge.execution import diff_prg, elwise_linear_prg
    knl = diff_prg(3, 1000000, 3, np.float64)
    params = dgk.run_tests.gen_autotune_list(queue, knl)

    args = [[param, knl] for param in params]

    # May help to balance workload
    from random import shuffle
    shuffle(args)
    
    #a = Array(AutotuneTask, dims=(len(args)), args=args[0])
    #a.get_queue()
   
    #result = charm.pool.map(do_work, args)

    pool_proxy = Chare(BalancedPoolScheduler, onPE=0)
    mypool = Pool(pool_proxy)
    result = mypool.map(do_work, args)

    sort_key = lambda entry: entry[0]
    result.sort(key=sort_key)
    

    for r in result:
        print(r)
"""

def main(args):
    import mpi4py.MPI as MPI
    from mirgecom.array_context import MirgecomAutotuningArrayContext as Maac
    #comm = MPI.COMM_WORLD
    
    autotune_pickled_kernels("./pickled_programs", 0, Maac, comm)
    print("DONE!")
    exit()

def charm_autotune():
    charm.start(main)
    print(result)
    charm.exit()
 
if __name__ == "__main__":
    charm.start(main)
    print(result)
    charm.exit()
