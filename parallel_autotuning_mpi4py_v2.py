#from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
#from charm4py.pool import PoolScheduler, Pool
#from charm4py.charm import Charm, CharmRemote
#from charm4py.chare import GROUP, MAINCHARE, ARRAY, CHARM_TYPES, Mainchare, Group, ArrayMap
#from charm4py.sections import SectionManager
#import inspect
#import sys
import hjson
import pyopencl as cl
import numpy as np
import os
#import grudge.grudge_array_context as gac
import loopy as lp
from os.path import exists
from run_tests import run_single_param_set_v2, generic_test
from utils import convert
#from grudge.execution import diff_prg, elwise_linear
import mpi4py.MPI as MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor
#from mpipool import MPIPool

#from guppy import hpy
import gc
import linecache
import os
import tracemalloc
#from mem_top import mem_top
#import matplotlib.pyplot as plt

data_dict = {}

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        d_str = filename + ":" + str(frame.lineno) + ": " + line
        if d_str not in data_dict:
            data_dict[d_str] = [stat.size]
        else:
            data_dict[d_str].append(stat.size)

        if line:
            print('    %s' % line)

    fig = plt.figure(0)
    fig.clear()
    plt.ion()
    plt.show()
    dlist = sorted(data_dict.items(), key=lambda a: a[1][-1], reverse=True)[:10]
    #print(dlist)
    #exit()
    for key, vals in dlist:
        plt.plot(vals, label=key + " " + str(vals[-1]) + " bytes")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=1)
    plt.draw()
    #plt.pause(1)
    plt.savefig("memory_usage.png", bbox_inches="tight")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


queue = None
def set_queue(pe_num, platform_num):
    global queue
    if queue is not None:
        raise ValueError("queue already set")
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Assume using platform zero
comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
# From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD
#queue = get_queue(comm.Get_rank(), 0)
#from feinsum.empirical_roofline import get_theoretical_maximum_flop_rate
#max_flop_rate = get_theoretical_maximum_flop_rate(queue, np.float64)

def test(args):
    global queue
    print(args)
    (cur_test, total_tests), (platform_id, knl, tlist, test_fn, max_flop_rate, device_latency, device_memory_bandwidth,) = args
    #comm = MPI.COMM_WORLD # Assume we're using COMM_WORLD. May need to change this in the future
    # From MPI.PoolExecutor the communicator for the tasks is not COMM_WORLD
    if queue is None:
        print("Queue is none. Initializing queue")
        set_queue(comm.Get_rank(), platform_id)
        assert queue is not None

    print(f"\nExecuting test {cur_test} of {total_tests}\n")
    result = run_single_param_set_v2(queue, knl, tlist, test_fn,
            max_flop_rate=max_flop_rate, 
            device_memory_bandwidth=device_memory_bandwidth,
            device_latency=device_latency)
    #print(mem_top())
    #h = hpy()
    #print(h.heap())
    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)
    #del knl
    #del args

    #result = [10,10,10]
    return result

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

            #del knl


def parallel_autotune(knl, platform_id, trans_list_list, program_id=None, max_flop_rate=None, device_latency=None, device_memory_bandwidth=None, save_path=None):

    if save_path is None:
        save_path = "./hjson"

    # Create queue, assume all GPUs on the machine are the same

    #comm = MPI.COMM_WORLD
    #platforms = cl.get_platforms()
    #gpu_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    #n_gpus = len(gpu_devices)
    #ctx = cl.Context(devices=[gpu_devices[comm.Get_rank() % n_gpus]])
    #profiling = cl.command_queue_properties.PROFILING_ENABLE
    #queue = cl.CommandQueue(ctx, properties=profiling)    

    #import pyopencl.tools as cl_tools
    #actx = actx_class(
    #    comm,
    #    queue,
    #    allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    #knl = gac.set_memory_layout(knl)
    if program_id is None:
        from utils import unique_program_id
        pid = unique_program_id(knl)
    else:
        pid = program_id

    #knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
    #knl = lp.set_options(knl, lp.Options(write_code=True))
    assert knl.default_entrypoint.options.no_numpy
    assert knl.default_entrypoint.options.return_dict
    #os.makedirs(os.getcwd() + "/hjson", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    hjson_file_str = f"{save_path}/{pid}.hjson"

    #assert comm.Get_size() > 1
    #assert charm.numPes() > 1
    #assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    #assert charm.numPes() <= charm.numHosts()*(len(gpu_devices) + 1)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes

    #if trans_list_list is None:
    #    tlist_generator = actx.get_generators(knl)
    #    trans_list_list = tlist_generator(actx.queue, knl)

    # Could make a massive list with all kernels and parameters
    ntransforms = len(trans_list_list)
    args = reversed([((ind+1,ntransforms),(platform_id, knl, tlist, generic_test, max_flop_rate, device_latency, device_memory_bandwidth,),) for ind, tlist in enumerate(trans_list_list)])

    #print(args)
    #exit()

    # May help to balance workload
    # Should test if shuffling matters
    #from random import shuffle
    #shuffle(args)


    #a = Array(AutotuneTask, dims=(len(args)), args=args[0])
    #a.get_queue()
   
    #result = charm.pool.map(do_work, args)

    #pool_proxy = Chare(BalancedPoolScheduler, onPE=0) # Need to use own charm++ branch to make work

    #pool_proxy = Chare(PoolScheduler, onPE=0)

    sort_key = lambda entry: entry["data"]["avg_time"]
    result = {"transformations": {}, "data": {}}
    comm = MPI.COMM_WORLD
    #nranks = comm.Get_size()
    if len(trans_list_list) > 0: # Guard against empty list
        #executor = MPIPoolExecutor(max_workers=1)
        #results = executor.map(test, args)
        #for entry in results:
        #    print(entry)
        #exit()
        #"""
        with MPICommExecutor(comm, root=0) as mypool:
            if mypool is not None:
                results = list(mypool.map(test, args, chunksize=1))
                results.sort(key=sort_key)
        
                # Workaround for pocl CUDA bug
                # whereby times are imprecise
                # There is a fix in a pull request for this
                ret_index = 0
                for i, result in enumerate(results):
                    if result["data"]["avg_time"] > 1e-7:
                        ret_index = i
                        break

                result = results[ret_index]
    
    #od = {"transformations": transformations, "data": data}
    if comm.Get_rank() == 0:
        print(result)
        out_file = open(hjson_file_str, "wt+")
        hjson.dump(result, out_file, default=convert)
        out_file.close()

    return result

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

def main():
    from mirgecom.array_context import MirgecomAutotuningArrayContext as Maac
    comm = MPI.COMM_WORLD
    
    tracemalloc.start()
    #gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
    autotune_pickled_kernels("./pickled_programs", 0, Maac, comm)

    print("DONE!")
    exit()

if __name__ == "__main__":
    import sys
    main()

    #pool = MPIPool()

    #if not pool.is_master():
    #    pool.wait()
    #    sys.exit(0)

