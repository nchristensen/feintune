from feintune.grudge_tags import (IsDOFArray, IsSepVecDOFArray, IsOpArray,
                                 IsSepVecOpArray, IsFaceDOFArray, IsFaceMassOpArray, IsVecDOFArray, IsVecOpArray, IsFourAxisDOFArray)
from feintune.apply_transformations import (gen_diff_knl, gen_diff_knl_fortran2,
                                           apply_transformation_list, gen_elwise_linear_knl, gen_face_mass_knl, gen_face_mass_knl_merged)
from feintune.utils import convert
import loopy.options
import sys
import time
import hjson
from pytools.obj_array import make_obj_array
import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from pyopencl.tools import ImmediateAllocator, MemoryPool
from immutabledict import immutabledict
# TODO Remove usage of Pebble, which is broken anyway.
from pebble import concurrent, ProcessExpired
from pebble.concurrent.process import _process_wrapper
from concurrent.futures import TimeoutError, BrokenExecutor
import multiprocessing as mp
import base64
from func_timeout import func_timeout, FunctionTimedOut
from multiprocessing import shared_memory

max_double = np.finfo('f').max
# from loopy.kernel.data import AddressSpace

"""
import pycuda.gpuarray as cuarray
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
"""

# from modepy import equidistant_nodes

# from math import ceil

# setup
# -----
lp.set_caching_enabled(False)
loopy.options.ALLOW_TERMINAL_COLORS = False

# import  grudge.grudge_array_context as gac#import set_memory_layout


def test_default_transforms(sk, queue, save_path=None, device_latency=None, device_memory_bandwidth=None, peak_flop_rate=None):

    if save_path is None:
        save_path = "kernel_tests_hjson"

    os.makedirs(save_path, exist_ok=True)

    # platforms = cl.get_platforms()
    # cl_ctx = cl.Context(
    #    dev_type=cl.device_type.GPU,
    #    properties=[(cl.context_properties.PLATFORM, platforms[0])])
    # queue = cl.CommandQueue(cl_ctx,
    #    properties=cl.command_queue_properties.PROFILING_ENABLE)

    from meshmode.array_context import PrefusedFusionContractorArrayContext
    actx = PrefusedFusionContractorArrayContext(queue)

    # device_latency=None
    # device_memory_bandwidth = None
    # clpeak_flop_rate = None

    # if profile_device:
    #    device_latency, device_memory_bandwidth, clpeak_flop_rate = get_device_roofline_data(queue)

    # gen_times = []

    for pid, sk in sk_list:
        # for sk in sk_list:
        # print(f"Testing subkernel: {pid}")

        einsum_counts = list(get_einsum_counts(sk).items())
        indirection = len(get_indirection_arrays(sk)) > 0
        if len(einsum_counts) > 0 and not indirection:
            # if len(einsum_counts) > 1:
            #    raise ValueError("Subkernel has multiple einsum types")

            einsum_type, einsum_count = einsum_counts[0]
            non_red_axes = len(einsum_type[0])
            red_axes = len(einsum_type[1])
            total_axes = non_red_axes + red_axes
            out_axes = total_axes - red_axes

            handled_pairs = set([(2, 1,), (3, 2,), (2, 2,), (2, 3)])
            # if True:
            if (non_red_axes, red_axes,) in handled_pairs and einsum_count > 0:

                start = time()
                # try:
                transformed_sk = actx.transform_loopy_program(sk)
                # except NotImplementedError:
                #    transformed_sk = sk
                # end = time()
                # transform_time = end - start
                # start = time()
                # """
                # """
                # code = lp.generate_code_v2(transformed_sk).device_code()
                # end = time()
                # codegen_time = end - start

                # name = transformed_sk.default_entrypoint.name
                # print(name, transform_time, codegen_time)

                # gen_times.append([name, transform_time, codegen_time])
                # """
                # """
                ret_dict = run_single_param_set_v2(queue, sk, [], generic_test,
                                                   max_flop_rate=peak_flop_rate, device_memory_bandwidth=device_memory_bandwidth,
                                                   device_latency=device_latency)

                # ret_dict = dict(ret_dict)
                # ret_dict["data"]["transform_time"] = transform_time
                # ret_dict["data"]["codegen_time"] = codegen_time
                # print(ret_dict["data"])
                # Should this functionality be a utility function
                hjson_file_str = save_path + f"/{pid}.hjson"
                out_file = open(hjson_file_str, "wt")
                hjson.dump(ret_dict, out_file, default=convert)
                out_file.close()

                # """
    # print("PRINTING RESULTS")
    # for name, transform_time, codegen_time in gen_times:
    #    print(name, transform_time, codegen_time)


def testBandwidth(fp_format=np.float32, nruns=100):

    from pyopencl.array import sum as clsum
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    # ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    from pyopencl.tools import ImmediateAllocator, MemoryPool
    allocator = ImmediateAllocator(queue)
    mem_pool = MemoryPool(allocator)

    knl = lp.make_copy_kernel("c,c", old_dim_tags="c,c")
    knl = lp.add_dtypes(knl, {"input": fp_format, "output": fp_format})
    # knl = knl.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
    n0 = 2
    # knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "i1", 256, inner_tag="l.0",
                         outer_tag="g.0", slabs=(0, 1))
    # knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0")
    # knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
    # knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))

    fp_bytes = 8 if fp_format == np.float64 else 4

    # This assumes fp32
    len_list = []
    float_count = 2
    max_floats = 2**28
    while float_count <= max_floats:
        len_list.append(float_count)
        float_count = int(np.ceil(float_count*1.5))
    for i in range(1, 29):
        len_list.append(2**i)
    len_list = sorted(list(set(len_list)))

    # data = np.random.randint(-127, 128, (1,max_bytes), dtype=np.int8)
    # inpt = cl.array.to_device(queue, data, allocator=mem_pool)

    print(len_list)

    for n in len_list:
        # for i in range(29):

        # n = 2**i
        kern = lp.fix_parameters(knl, n0=n0, n1=n)
        # data = np.random.randint(-127, 128, (1,n), dtype=np.int8)
        # inpt = cl.array.to_device(queue, data, allocator=mem_pool)
        inpt = cl.clrandom.rand(queue, (n0, n), dtype=fp_format)
        outpt = cl.array.Array(
            queue, (n0, n), dtype=fp_format, allocator=mem_pool)

        # Output code before editing it
        kern = lp.set_options(kern, "write_code")

        for j in range(2):
            kern(queue, input=inpt, output=outpt)
        dt = 0
        events = []
        for j in range(nruns):
            evt, _ = kern(queue, input=inpt, output=outpt)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            dt += evt.profile.end - evt.profile.start
        # queue.finish()
        dt = dt / nruns / 1e9

        nbytes_transferred = 2*fp_bytes*n*n0
        bandwidth = nbytes_transferred / dt / 1e9
        print("{} {}".format(nbytes_transferred, bandwidth))

        # print((inpt - outpt))
        # diff = (inpt - outpt)
        # if  clsum(inpt - outpt) != 0:
        #    print("INCORRECT COPY")


def test_face_mass_merged(kern, backend="OPENCL", nruns=10, warmup=True):
    # kern = gen_diff_knl(n_elem, n_in, n_out, k_inner_outer, k_inner_inner,
    #    i_inner_outer, i_inner_inner, j_inner)
    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")
    for arg in kern.args:
        if arg.name == "vec":
            fp_format = arg.dtype
            n_elem, n_in = arg.shape
        elif arg.name == "mat":
            n_out, _ = arg.shape

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        print("Not supported")
        exit()
    elif OPENCL:
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(
            device_type=cl.device_type.GPU)
        # ctx = cl.Context(devices=my_gpu_devices)
        ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        # Output code before editing it
        kern = lp.set_options(kern, "write_code")
        # Print the Code
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]  # .decode(
        # errors="ignore") #Breaks pocl
        from bs4 import UnicodeDammit
        dammit = UnicodeDammit(ptx)
        # print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()

        from pyopencl.tools import ImmediateAllocator, MemoryPool
        allocator = ImmediateAllocator(queue)
        mem_pool = MemoryPool(allocator)

        X_dev = cl.array.Array(queue, (n_elem, n_in),
                               dtype=fp_format, order="F", allocator=mem_pool)
        cl.clrandom.fill_rand(X_dev, queue=queue)
        B_dev = cl.array.Array(queue, (n_elem, n_out),
                               dtype=fp_format, allocator=mem_pool, order="F")
        A_dev = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)

        if warmup:
            for i in range(2):
                kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            queue.finish()

        sum_time = 0.0
        events = []
        for i in range(nruns):
            evt, _ = kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9
        # queue.finish()

    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time


def measure_execution_time(queue, tunit, arg_dict, nruns, warmup_runs, pollute_buffers=(None, None,)):

    in_pollute, out_pollute = pollute_buffers
    pollute_caches = True if in_pollute is not None and out_pollute is not None else False

    print("Warming up")
    for i in range(warmup_runs):
        print("Warmup run", i)
        if pollute_caches:
            cl.enqueue_copy(queue, out_pollute, in_pollute)
        tunit(queue, **arg_dict)
    print("Done warming up")
    # queue.finish()


    sum_time = 0.0
    events = []
    # Should the cache be polluted between runs?
    print("Executing")
    for i in range(nruns):
        if pollute_caches:
            cl.enqueue_copy(queue, out_pollute, in_pollute)
        evt, out = tunit(queue, **arg_dict)
        events.append(evt)
    queue.finish()
    cl.wait_for_events(events)
    for evt in events:
        sum_time += evt.profile.end - evt.profile.start

    avg_time = sum_time / 1e9 / nruns
    return avg_time


# Strips out instructions and executes kernel to see how long the
# argument setting, etc, requires. This assumes there is only
# one kernel or function in the generated code
def measure_execution_latency(queue, tunit, arg_dict, nruns, warmup_runs):
    print("Starting measuring latency")
    args = arg_dict.items()
    arg_names = [entry[0] for entry in args]
    arg_vals = [entry[1].data for entry in args]

    otunit = lp.set_argument_order(tunit, arg_names)
    code = lp.generate_code_v2(otunit).device_code()
    from feintune.matching_brackets import matching_brackets_dict
    fn_brackets = sorted(matching_brackets_dict(
        code, opening_bracket="{", closing_bracket="}").items(), key=lambda l: l[1], reverse=True)[0]

    null_kernel_code = code[:fn_brackets[0] + 1] + code[fn_brackets[1]:]
    # null_kernel_code = code.split("{")[0] + "{}"
    search_str = "reqd_work_group_size("
    start_ind = null_kernel_code.index(search_str) + len(search_str)
    sub_str = null_kernel_code[start_ind:].split(",")

    lwork_size = []
    for i in range(3):
        lwork_size.append(np.int32(sub_str[i].split(")")[0].replace(" ", "")))

    # print(null_kernel_code)
    # print(lwork_size)

    program = cl.Program(queue.context, null_kernel_code).build()
    cl_knl = program.all_kernels()[0]
    # nargs = cl_knl.num_args
    # name_to_ind = {cl_knl.get_arg_info(ind, cl.kernel_arg_info.NAME): ind for ind in range(nargs)}

    cl_knl.set_args(*arg_vals)
    # for key, val in arg_dict.items():
    #    ind = name_to_ind[key]
    #    cl_knl.set_arg(ind, val)

    print("Launching null kernel")
    for i in range(warmup_runs):
        cl.enqueue_nd_range_kernel(queue, cl_knl, lwork_size, lwork_size)

    events = []
    for i in range(warmup_runs):
        events.append(cl.enqueue_nd_range_kernel(
            queue, cl_knl, lwork_size, lwork_size))
    queue.finish()
    cl.wait_for_events(events)
    sum_time = 0.0
    min_latency = np.inf
    for evt in events:
        lat_val = evt.profile.end - evt.profile.start
        sum_time += lat_val
        if lat_val < min_latency:
            min_latency = lat_val

    min_latency = min_latency / 1e9
    avg_latency = sum_time / 1e9 / nruns
    print("Finished measuring latency")
    return min_latency


# Maybe the queue could also be a cuda stream? Could use the type of that to
# distinguish between CUDA and OpenCL possibly
# This hardcodes the memory layout, should probably instead retrieve it from somewhere on a per
# tag basis

# cache_arg_dict = {}
# HIPBLAS seeems to perform fine after one warmup round. Could probably
# set warmup_runs to 2.
def generic_test(queue, kern, backend="OPENCL", nruns=10, warmup_runs=2, measure_latency=True):

    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        # Scrounge up code in the "nick" repository on Andreas's gitlab if
        # desire to implement this.
        print("CUDA not supported")
        exit()
    elif OPENCL:
        """
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        #ctx = cl.create_some_context(interactive=True)
        #queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it
        # Print the Code
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        dammit = UnicodeDammit(ptx)
        print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()
        """

        print("STARTING ALLOCATION")
        start = time.time()
        allocator = ImmediateAllocator(queue)
        mem_pool = MemoryPool(allocator)
        # mem_pool = get_reasonable_memory_pool(queue)
        # print("USING MEMORY POOL OF TYPE", type(mem_pool))
        # exit()

        arg_dict = {}

        # Fill arrays with random data
        # Could probably just read the strides from the kernel to get ordering
        for arg in [filt_arg for filt_arg in  kern.default_entrypoint.args if isinstance(filt_arg, lp.ArrayArg)]:
            if True:  # str(arg) not in cache_arg_dict:
                # print(arg)
                fp_bytes = arg.dtype.numpy_dtype.itemsize
                if IsDOFArray() in arg.tags:
                    array = cl.array.Array(
                        queue, arg.shape, arg.dtype, order="F", allocator=mem_pool)
                    # if not arg.is_output:
                    #    cl.clrandom.fill_rand(array, queue)
                elif IsSepVecDOFArray() in arg.tags:
                    # if arg.is_output:
                    obj_array = [cl.array.Array(
                        queue, arg.shape[1:], dtype=arg.dtype, allocator=mem_pool, order="F") for i in range(arg.shape[0])]
                    array = make_obj_array(obj_array)
                    # else:
                    #    print("Input SepVecDOFArrays are not currently supported")
                    #    exit()
                elif IsFaceDOFArray() in arg.tags:
                    # fp_bytes = arg.dtype.numpy_dtype.itemsize
                    nfaces, nelements, nface_nodes = arg.shape
                    strides = (fp_bytes*nelements, fp_bytes*1,
                               fp_bytes*nelements*nfaces)  # original
                    array = cl.array.Array(queue, arg.shape, dtype=arg.dtype,
                                           strides=strides, allocator=mem_pool)
                    # cl.clrandom.fill_rand(array, queue=queue)
                elif IsVecDOFArray() in arg.tags:
                    # fp_bytes = arg.dtype.numpy_dtype.itemsize
                    nr, nelements, ndofs = arg.shape
                    strides = (fp_bytes*nelements*ndofs, fp_bytes,
                               fp_bytes*nelements)  # original
                    array = cl.array.Array(queue, arg.shape, dtype=arg.dtype,
                                           strides=strides, allocator=mem_pool)
                    # cl.clrandom.fill_rand(array, queue=queue)
                elif IsFourAxisDOFArray() in arg.tags:
                    # fp_bytes = arg.dtype.numpy_dtype.itemsize
                    nx, nr, nelements, ndofs = arg.shape
                    strides = (fp_bytes*nelements*ndofs*nr, fp_bytes*nelements*ndofs,
                               fp_bytes, fp_bytes*nelements)
                    array = cl.array.Array(queue, arg.shape, dtype=arg.dtype,
                                           strides=strides, allocator=mem_pool)
                    # cl.clrandom.fill_rand(array, queue=queue)
                elif IsSepVecOpArray() in arg.tags:
                    # obj_array = [cl.clrandom.rand(queue, arg.shape[1:], dtype=arg.dtype) for i in range(arg.shape[0])]
                    obj_array = [cl.array.Array(
                        queue, arg.shape[1:], dtype=arg.dtype, order="C", allocator=mem_pool) for i in range(arg.shape[0])]
                    # if not arg.is_output:
                    #    cl.clrandom.fill_rand(arg_dict[arg.name], queue)
                    # obj_array = []
                    # for i in range(arg.shape[0]):
                    # clarray = cl.array.Array(queue, arg.shape[1:], arg.dtype, order="C", allocator=mem_pool)
                    # cl.clrandom.fill_rand(clarray, queue=queue)
                    # obj_array.append(clarray)
                    array = make_obj_array(obj_array)
                # elif IsFaceMassOpArray() in arg.tags:
                    # Are these strides correct?
                    # array = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)
                elif IsOpArray() in arg.tags or IsVecOpArray() in arg.tags or IsFaceMassOpArray in arg.tags:
                    array = cl.array.Array(
                        queue, arg.shape, arg.dtype, order="C", allocator=mem_pool)
                    # cl.clrandom.fill_rand(array, queue=queue)
                    # array = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)
                elif isinstance(arg, lp.ArrayArg):
                    array = cl.array.Array(
                        queue, arg.shape, arg.dtype, order="C", allocator=mem_pool)
                    # cl.clrandom.fill_rand(array, queue=queue)
                    print(arg.name, "No tags recognized. Assuming default data layout")
                    # Assume default layout
                    # array = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)

                if not arg.is_output:
                    if isinstance(array, cl.array.Array):
                        if array.dtype == np.int8:
                            # Could generalize this for all unhandled dtypes
                            npa = np.random.randint(-128, high=127, size=array.shape, dtype=array.dtype)
                            array.set(npa, queue=queue)
                        else:
                            cl.clrandom.fill_rand(array, queue=queue)
                    elif isinstance(array[0], cl.array.Array):
                        for entry in array:
                            cl.clrandom.fill_rand(entry, queue=queue)
                    else:
                        raise TypeError

                # cache_arg_dict[str(arg)] = array
                # print(arg.name)
                # print(arg.tags)
                # print("Unknown Tag")
                # exit()

            # arg_dict[arg.name] = cache_arg_dict[str(arg)]
            arg_dict[arg.name] = array
            end = time.time()
        print("ENDING ALLOCATION", end - start, "seconds")
        print("STARTING EXECUTION")
        start = time.time()

        # print("Setting measured execution latency to zero")
        measured_latency = None
        if measure_latency:
            try:
                measured_latency = measure_execution_latency(
                    queue, kern, arg_dict, nruns, warmup_runs)
            except Exception as e:
                print("Unable to measure null kernel latency due to error.")
                print(e)

        pollute_size = (10*queue.device.global_mem_cache_size) // 4
        if pollute_size == 0:
            # Pocl CUDA doesn't currently report global memory cache information
            # Presumably it isn't that much different from the local memory size, so
            # use that instead.
            pollute_size = (10*queue.device.local_mem_size) // 4

        from feintune.empirical_roofline import get_buffers
        d_in_buf, d_out_buf = get_buffers(queue, np.int32, pollute_size, dtype_out=np.int32, n_dtype_out=pollute_size, fill_on_device=True)

        avg_time = measure_execution_time(
                queue, kern, arg_dict, nruns, warmup_runs, pollute_buffers=(d_in_buf, d_out_buf))

        end = time.time()
        print("FINISHING EXECUTION", end - start, "seconds")

        # queue.finish()
    # sum_time = 1.0

    return arg_dict, avg_time, measured_latency


def get_knl_device_memory_bytes(knl):

    # What if the output is not in the input arguments?
    # print(knl.default_entrypoint.args)
    # Would probably be better to use the memory footprint
    # if can get it to work.

    args_and_temps = knl.default_entrypoint.args + \
        list(knl.default_entrypoint.temporary_variables.values())

    read_deps = set()
    write_deps = set()
    for instr in knl.default_entrypoint.instructions:
        read_deps |= instr.read_dependency_names()
        write_deps |= instr.write_dependency_names()

    nbytes = 0
    for arg in args_and_temps:
        if isinstance(arg, lp.ArrayArg) and  arg.address_space == lp.AddressSpace.GLOBAL:
            if arg.name in read_deps:
                nbytes += np.prod((arg.shape))*arg.dtype.dtype.itemsize
            if arg.name in write_deps:
                nbytes += np.prod((arg.shape))*arg.dtype.dtype.itemsize

    return nbytes


# avg_time in seconds
def analyze_knl_bandwidth(knl, avg_time, device_latency=None):
    # This bandwidth calculation assumes data in global memory need only be accessed once
    # from global memory and is otherwise served from a cache or local memory that is
    # fast enough to be considered free
    if device_latency is None:
        device_latency = 0
    nbytes = get_knl_device_memory_bytes(knl)
    bw = nbytes / (avg_time - device_latency)

    # Seems lp.gather_access_footprint_bytes breaks
    # footprint = lp.gather_access_footprint_bytes(knl)
    # footprint_bytes = 0
    # for val in footprint.values():
    #    footprint_bytes += val.eval_with_dict({})
    # footprint_bw =  footprint_bytes / avg_time / 1e9
    # print(f"Time: {avg_time}, Bytes: {nbytes}, Bandwidth: {bw} GB/s Footprint BW: {footprint_bw} GB/s")
    Gbps = bw*1e-9

    print(f"Time: {avg_time}, Bytes: {nbytes}, Bandwidth: {Gbps} GB/s")
    return immutabledict({"observed_bandwidth": bw,
                       "nbytes_global": nbytes,
                       "device_latency": device_latency})


def get_knl_flops(tunit):

    tunit = tunit.with_kernel(tunit.default_entrypoint.copy(silenced_warnings=(tunit.default_entrypoint.silenced_warnings
                                                                               + ["insn_count_subgroups_upper_bound",
                                                                                  "summing_if_branches_ops", "count_overestimate"])))

    # There is a more complex version of this in meshmode.arraycontext
    try:
        op_map = lp.get_op_map(
            tunit, count_within_subscripts=False, subgroup_size=1)
    except AssertionError:
        # For some kernels, lp.get_op_map fails
        return -1

    map_flops = 0
    for val in op_map.values():
        map_flops += val.eval_with_dict({})
    return map_flops


# Avg time in seconds, max_flop_rate in flops per second
def analyze_flop_rate(knl, avg_time, max_flop_rate=None, latency=None):
    map_flops = get_knl_flops(knl)
    flop_rate = map_flops / avg_time
    if latency is None:
        latency = 0

    """
    n_mat = 1
    nfaces = 1
    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsSepVecOpArray() in arg.tags or IsVecOpArray() in arg.tags:
            n_mat, n_out, n_in = arg.shape
        elif IsOpArray() in arg.tags:
            n_out, n_in = arg.shape
        elif IsFaceDOFArray() in arg.tags:
            nfaces, n_elem, n_in = arg.shape
    
    flops = nfaces*n_mat*2*(n_out * n_in * n_elem)
    """
    assert latency >= 0
    assert avg_time - latency > 0
    # Subtract memory latency from flop_rate if known
    flop_rate = map_flops / (avg_time - latency)
    print("GFLOP/s: " + str(flop_rate*1e-9))

    # print("Map GFLOP/s: " + str(map_gflop_rate))
    # print(flops)
    # print(map_flops)
    """
    frac_peak_gflops = None
    if max_gflops is not None:
        print("Peak GFLOP/s: " + str(max_gflops))
        frac_peak_gflops = gflop_rate / max_gflops
        print("Percent peak: " + str(100*(frac_peak_gflops)))
    """

    # Calculate bandwidth
    # Assumes each element only read once
    # ideal_total_bytes_transferred = fp_bytes*(3*(n_out * n_elem) + (n_in * n_elem)
    #                                            + 3*(n_out * n_in))
    # GBps = (ideal_total_bytes_transferred / avg_time) / 1e9
    # frac_peak_GBps = GBps / device_memory_bandwidth
    # print("GB/s: " + str(GBps))
    # print("Peak GB/s: " + str(device_memory_bandwidth))
    # print("Percent peak: " + str(100*(frac_peak_GBps)))
    # print()

    # gflop_rate, frac_peak_gflops
    return immutabledict({"observed_flop_rate": flop_rate, "flops": map_flops})


def get_knl_device_memory_roofline(knl, max_flop_rate, device_latency, device_memory_bandwidth):
    device_memory_bytes = get_knl_device_memory_bytes(knl)
    flops_per_byte = get_knl_flops(knl) / device_memory_bytes
    effective_bandwidth = device_memory_bytes / \
        (device_latency + device_memory_bytes / device_memory_bandwidth)
    roofline_flop_rate = min(flops_per_byte*effective_bandwidth, max_flop_rate)
    return roofline_flop_rate


def verifyResult(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev):
    A_host1 = A_dev1.get()
    A_host2 = A_dev2.get()
    A_host3 = A_dev3.get()
    X_host = X_dev.get()
    B_host1 = B_dev1.get()
    B_host2 = B_dev2.get()
    B_host3 = B_dev3.get()
    np.set_printoptions(threshold=sys.maxsize)
    errMat = ((A_host1 @ X_host) - B_host1) / np.linalg.norm(A_host1 @ X_host)
    print("Fraction Nonzero: " + str(np.count_nonzero(errMat)/(n_out*n_elem)))
    print("Norm1: " + str(np.linalg.norm((A_host1 @ X_host) - B_host1)
                          / np.linalg.norm(A_host1 @ X_host)))
    print("Norm2: " + str(np.linalg.norm((A_host2 @ X_host) - B_host2)
                          / np.linalg.norm(A_host2 @ X_host)))
    print("Norm3: " + str(np.linalg.norm((A_host3 @ X_host) - B_host3)
                          / np.linalg.norm(A_host3 @ X_host)))


def verifyResultFortran(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev):
    A_host1 = A_dev1.get()
    A_host2 = A_dev2.get()
    A_host3 = A_dev3.get()
    X_host = X_dev.get().T
    B_host1 = B_dev1.get()
    B_host2 = B_dev2.get()
    B_host3 = B_dev3.get()
    np.set_printoptions(threshold=sys.maxsize)
    errMat = ((A_host1 @ X_host).T - B_host1) / \
        np.linalg.norm(A_host1 @ X_host)
    print("Fraction Nonzero: " + str(np.count_nonzero(errMat)/(n_out*n_elem)))
    print("Norm1: " + str(np.linalg.norm((A_host1 @ X_host).T - B_host1)
                          / np.linalg.norm(A_host1 @ X_host)))
    print("Norm2: " + str(np.linalg.norm((A_host2 @ X_host).T - B_host2)
                          / np.linalg.norm(A_host2 @ X_host)))
    print("Norm3: " + str(np.linalg.norm((A_host3 @ X_host).T - B_host3)
                          / np.linalg.norm(A_host3 @ X_host)))


# This can be removed eventually
def apply_transformations_and_run_test(queue, knl, test_fn, params, tgenerator, max_gflops=None,
                                       device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):

    kio, kii, iio, iii, ji = params

    # Transform and run
    # knl = gac.set_memory_layout(knl)
    if applicator is not None:
        trans_list = tgenerator(params)
    else:
        # Should probably read in eligible transformations from a file instead of using if-statements
        trans_list = []
        if "diff" in knl.default_entrypoint.name:
            trans_list.append(["tag_inames", ["imatrix: ilp"]])

        trans_list.append(["split_iname", ["iel", kio], {
                          "outer_tag": "g.0", "slabs": (0, 1)}])
        trans_list.append(["split_iname", ["iel_inner", kii],
                           {"outer_tag": "ilp", "inner_tag": "l.0", "slabs": (0, 1)}])
        trans_list.append(["split_iname", ["idof", iio], {
                          "outer_tag": "g.1", "slabs": (0, 0)}])
        trans_list.append(["split_iname", ["idof_inner", iii],
                           {"outer_tag": "ilp", "inner_tag": "l.1", "slabs": (0, 1)}])

        if knl.default_entrypoint.name == "face_mass":
            pass
            # trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
            #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
            # trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
        elif knl.default_entrypoint.name == "nodes":
            trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                               {"temporary_name": "vecf", "default_tag": "l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
        elif "resample_by_mat" in knl.default_entrypoint.name:
            # Indirection may prevent prefetching
            pass
        else:
            trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                               {"temporary_name": "vecf", "default_tag": "l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

        trans_list.append(["split_iname", ["j", ji], {
                          "outer_tag": "for", "inner_tag": "for"}])
        trans_list.append(["add_inames_for_unused_hw_axes"])

    knl = apply_transformation_list(knl, trans_list)

    # print(knl.default_entrypoint.name)
    # print(trans_list)

    # Execute and analyze the results
    dev_arrays, avg_time = test_fn(queue, knl)
    # avg_time = np.random.rand()

    return avg_time, trans_list

    """
    # The analysis should be done elsewhere
    bw = None
    flop_rate = None

    if device_memory_bandwidth is not None:  # noqa
	bw = analyze_knl_bandwidth(knl, avg_time)
	frac_peak_GBps = bw / device_memory_bandwidth
	if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
	    # Should validate result here
	    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
	    return avg_time, params

    # Einsum complicates this. This depends on the kernel being called.
    if max_gflops is not None:
	frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
	if frac_peak_gflops >= gflops_cutoff:
	    # Should validate result here
	    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
	    return choices

    if device_memory_bandwidth is not None and max_gflops is not None:
	data = (avg_time, 
			    frac_peak_GBps*device_memory_bandwidth, 
			    frac_peak_gflops*max_gflops, 
			    frac_peak_GBps, 
			    frac_peak_gflops, 
			    (kio, kii, iio, iii, ji))
	result_list.append(data)
	f.write(str(data) + "\n")

    if avg_time < avg_time_saved:
	avg_time_saved = avg_time
	result_saved = choices
	result_saved_list = trans_list
    if time.time() - start > time_limit: 
	result_list.sort()
	print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
	for entry in result_list:
	    print(entry)
	print()


	#return result_saved_list
	return result_saved
    """


def run_single_param_set(queue, knl_base, tlist_generator, params, test_fn, max_gflops=None, device_memory_bandwidth=None):
    trans_list = tlist_generator(params, knl=knl_base)
    knl = apply_transformation_list(knl_base, trans_list)
    dev_arrays, avg_time = test_fn(queue, knl)

    # Should this return the fraction of peak of should that be calculated in this function?
    gflops, frac_peak_gflops = analyze_FLOPS(
        knl, avg_time, max_gflops=max_gflops)
    bw = analyze_knl_bandwidth(knl, avg_time)

    if device_memory_bandwidth is not None:  # noqa
        # bw = analyze_knl_bandwidth(knl, avg_time)
        frac_peak_GBps = bw / device_memory_bandwidth
        if frac_peak_GBps >= bandwidth_cutoff:  # noqa
            # Should validate result here
            print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
            return choices

    # This is incorrect for general einsum kernels
    if max_gflops is not None:
        # analyze_FLOPS(knl, max_gflops, avg_time)
        frac_peak_gflops = gflops / max_gflops
        if frac_peak_gflops >= gflops_cutoff:
            # Should validate result here
            print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
            return choices

    data = {"avg_time": avg_time, "observed_GBps": bw,
            "observed_gflop_rate": gflops}
    if device_memory_bandwidth is not None and max_gflops is not None:
        data.update({"max_gflops": max_gflops,
                     "device_memory_GBps": device_memory_bandwidth,
                     "frac_peak_GBps": frac_peak_GBps,
                     "frac_peak_gflops": frac_peak_gflops
                     })

    retval = {"transformations": trans_list, "data": data}
    return retval


# Ripped from mirgecom
def get_reasonable_memory_pool(queue: cl.CommandQueue,
                               force_buffer: bool = False,
                               force_non_pool: bool = False):
    """Return an SVM or buffer memory pool based on what the device supports.

    By default, it prefers SVM allocations over CL buffers, and memory
    pools over direct allocations.
    """
    import pyopencl.tools as cl_tools
    from pyopencl.characterize import has_coarse_grain_buffer_svm
    ctx = queue.context

    if force_buffer and force_non_pool:
        # logger.info(f"Using non-pooled CL buffer allocations on {queue.device}.")
        return cl_tools.DeferredAllocator(ctx)

    if force_buffer:
        # logger.info(f"Using pooled CL buffer allocations on {queue.device}.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))

    if force_non_pool and has_coarse_grain_buffer_svm(queue.device):
        # logger.info(f"Using non-pooled SVM allocations on {queue.device}.")
        return cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue)

    if has_coarse_grain_buffer_svm(queue.device) and hasattr(cl_tools, "SVMPool"):
        # logger.info(f"Using SVM-based memory pool on {queue.device}.")
        return cl_tools.SVMPool(cl_tools.SVMAllocator(  # pylint: disable=no-member
            ctx, alignment=0, queue=queue))
    else:
        from warnings import warn
        if not has_coarse_grain_buffer_svm(queue.device):
            warn(f"No SVM support on {queue.device}, returning a CL buffer-based "
                 "memory pool. If you are running with PoCL-cuda, please update "
                 "your PoCL installation.")
        else:
            warn("No SVM memory pool support with your version of PyOpenCL, "
                 f"returning a CL buffer-based memory pool on {queue.device}. "
                 "Please update your PyOpenCL version.")
        return cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))


# Useful elsewhere. Maybe move to utils or as a utility function in pyopencl
def get_queue_from_bus_id(bus_id, platform_name=None):

    if platform_name is None:
        platforms = cl.get_platforms()
    else: 
        platforms = [platform for platform in cl.get_platforms() if platform.name == platform_name]

    for platform in platforms:
        for d in platform.get_devices():
            if "NVIDIA" in d.vendor:
                d_bus_id = d.pci_bus_id_nv
            elif "Advanced Micro Devices" in d.vendor:
                d_bus_id = d.topology_amd.bus
            else:
                d_bus_id = None

            if d_bus_id == bus_id:
                ctx = cl.Context(devices=[d])
                my_queue = cl.CommandQueue(
                    ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return my_queue

    raise RuntimeError(f"No device with bus_id {bus_id} found")


# Useful elsewhere. Maybe move to utils or as a utility function in pyopencl
# Should this also return the platform id?
def get_bus_id_from_queue(queue):
    d = queue.device
    if "NVIDIA" in d.vendor:
        d_bus_id = d.pci_bus_id_nv
    elif "Advanced Micro Devices" in d.vendor:
        d_bus_id = d.topology_amd.bus
    else:
        raise RuntimeError("Can't query bus id")
    return d_bus_id


def run_subprocess_with_timeout(queue, knl, test_fn, timeout=max_double, error_return_time=max_double):

    start = time.time()

    import os
    import uuid
    from pickle import dump, dumps
    from subprocess import run, TimeoutExpired, CalledProcessError, Popen, PIPE, STDOUT
    import feintune
    #import logging

    #from ytopt.search import util

    #logger = util.conf_logger('feintune.run_tests')



    # pickled_knl = base64.b85encode(dumps(knl)).decode('ASCII')
    # pickled_test_fn = base64.b85encode(dumps(test_fn)).decode('ASCII')
    bus_id = get_bus_id_from_queue(queue)
    platform_name = queue.device.platform.name

    # filename = str(uuid.uuid4()) + ".tmp"
    # out_file = open(filename, "wb")
    # dump(tuple([knl, test_fn, bus_id]), out_file)
    # out_file.close()

    pickled_data = dumps([knl, test_fn, bus_id, platform_name])
    shm = shared_memory.SharedMemory(create=True, size=len(pickled_data))
    shm.buf[:] = pickled_data[:]

    #"""
    dirname = os.path.dirname(feintune.__file__)
    f = os.path.join(dirname, "run_tests.py")

    stderr_file = open("./stderr_file.txt", 'wt')

    proc = Popen(["python", f, shm.name], stdout=PIPE,
                 stderr=stderr_file, text=True)#, env=os.environ)
    try:
        output, err = proc.communicate(timeout=timeout)
        #logger.info("SUBPROCESS OUTPUT", output)
        #logger.info("SUBPROCESS ERR", err)
        if proc.returncode != 0:
            raise CalledProcessError(proc.returncode, proc.args, output=output)
        split_output = output.split("|")
        end = time.time()
        retval = float(split_output[-3]), float(split_output[-1]), end - start
    except TimeoutExpired:
        print("Subprocess timed out")
        proc.kill()
        # with proc:
        #    proc.kill()
        retval = error_return_time, None, 0

    # out, err = proc.communicate()

    # try:
    #    start = time.time()
    #    #completed = run(["python", "run_tests.py", str(bus_id), pickled_knl, pickled_test_fn],
    #    #                capture_output=True, check=True, timeout=timeout, text=True)

    #    end = time.time()
    #    output = completed.stdout
    #    split_output = output.split("|")
    #    retval = float(split_output[-3]), float(split_output[-1]), end - start
    # except TimeoutExpired as e:
    #    print("Subprocess timed out")
    #    return max_double, max_double, 0
    except CalledProcessError as e:
        print("Subprocess failed with the following output:")
        print(e.output)
        # os.remove(filename)
        proc.kill()
        retval = error_return_time, None, 0
        # shm.unlink()
        exit()

    shm.close()
    shm.unlink()
    #"""

    #avg_time, measured_latency = unpickle_and_run_test(shm.name)
    #end = time.time()
    #retval = avg_time, measured_latency, end - start


    # os.remove(filename)

    return retval


def unpickle_and_run_test(sh_mem_name):
    from pickle import loads
    import sys
    from multiprocessing.resource_tracker import unregister

    sh_mem = shared_memory.SharedMemory(sh_mem_name)
    knl, test_fn, bus_id, platform_name = loads(sh_mem.buf)
    sh_mem.close()

    #sh_mem.unlink() # Doing this in the main process instead

    # Workaround for https://github.com/python/cpython/issues/82300
    # Hopefully will be fixed in 3.13
    if shared_memory._USE_POSIX and sys.version_info <= (3, 12):
        unregister(sh_mem._name, "shared_memory")

    # in_file = open(filename, "rb")
    # knl, test_fn, bus_id = load(in_file)
    # in_file.close()

    # knl = loads(base64.b85decode(pickled_knl.encode('ASCII')))
    # test_fn = loads(base64.b85decode(pickled_test_fn.encode('ASCII')))
    queue = get_queue_from_bus_id(int(bus_id), platform_name=platform_name)
    dev_arrays, avg_time, measured_latency = test_fn(queue, knl)

    # Alternatively, could write this back to the shared memory.
    print("|Average execution time|", avg_time,
          "|Average execution latency|", measured_latency)

    return avg_time, measured_latency

"""
def unpickle_and_run_test(filename):
    from pickle import load

    in_file = open(filename, "rb")
    knl, test_fn, bus_id = load(in_file)
    in_file.close()
    
    #knl = loads(base64.b85decode(pickled_knl.encode('ASCII')))
    #test_fn = loads(base64.b85decode(pickled_test_fn.encode('ASCII')))
    queue = get_queue_from_bus_id(int(bus_id))
    dev_arrays, avg_time, measured_latency = test_fn(queue, knl)
    
    print("|Average execution time|", avg_time, "|Average execution latency|", measured_latency)
"""

# Need to change the timeout so can't use this decorate (and can't decorating
# a function while using 'spawn' is not support inside another function)
# @concurrent.process(context=mp.get_context('spawn'), timeout=autotune_timeout)


def test_fn_wrapper(bus_id, knl, test_fn):
    queue = get_queue_from_bus_id(bus_id)
    dev_arrays, avg_time, measured_latency = test_fn(queue, knl)
    return avg_time, measured_latency


# mp_context = mp.get_context('spawn')
mp_context = mp.get_context('forkserver')


def run_concurrent_test_with_timeout(queue, knl, test_fn, timeout=None, method="threadpool"):

    # Cuda initialization fails with fork, but spawn and forkserver may also not play well with mpi

    bus_id = get_bus_id_from_queue(queue)

    if method == "pebble":
        wrapped_fn = _process_wrapper(
            test_fn_wrapper, timeout, None, None, mp_context)
        future = wrapped_fn(bus_id, knl, test_fn)
        executor = None
    else:
        if method == "processpool":
            from concurrent.futures import ProcessPoolExecutor as Executor
            executor = Executor(max_workers=1, mp_context=mp_context)
        else:
            from concurrent.futures import ThreadPoolExecutor as Executor
            executor = Executor(max_workers=1)

        future = executor.submit(test_fn_wrapper, bus_id, knl, test_fn)

    start = time.time()
    try:
        avg_time, measured_latency = future.result(timeout=timeout)
    except TimeoutError as error:
        print("Test function timed out. Time limit %f seconds. Returning null result" %
              float(error.args[1]))
        avg_time, measured_latency = max_double, 0
    except BrokenExecutor as error:
        print("Executor broke. This may be due to the GPU code crashing the process.")
        print(error)
        avg_time, measured_latency = max_double, 0
    except ProcessExpired as error:
        print("%s. Exit code: %d" % (error, error.exitcode))
        avg_time, measured_latency = max_double, 0
    except Exception as error:
        print("Test function raised %s" % error)
        print(error.traceback)  # traceback of the function
    end = time.time()

    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)

    wall_clock_time = end - start if avg_time != max_double else max_double

    return avg_time, measured_latency, wall_clock_time


# , method="thread"):
def run_single_param_set_v2(queue, knl_base, trans_list, test_fn, max_flop_rate=None, device_memory_bandwidth=None, device_latency=None, timeout=None, method=None, run_single_batch=False, error_return_time=None, measure_latency=True):

    if measure_latency == False and method is not None:
        # Haven't yet passed this parameter
        raise NotImplementedError

    # Should check how well single batch predicted times correllate with actual times

    #if device_latency is None:
    #    device_latency = 0
    #if max_flop_rate is None:
    #    max_flop_rate = np.inf
    #if device_memory_bandwidth is None:
    #    device_memory_bandwidth = np.inf

    if error_return_time is None:
        error_return_time = timeout + 1 if timeout is not None else max_double

    from feintune.apply_transformations import get_einsums
    neinsums = len(get_einsums(knl_base))
    batch_size = neinsums  # No batching is equivalent to one batch

    print("BEGINNING KERNEL TRANSFORMATION")

    try:
        if timeout is None:
            knl, sb_knl = apply_transformation_list(knl_base, trans_list)
            knl = lp.preprocess_kernel(knl)
            insn_ids = tuple(
                [insn.id for insn in knl.default_entrypoint.instructions])
            group_sizes, local_sizes = knl.default_entrypoint.get_grid_sizes_for_insn_ids(
                insn_ids, None)
            transformed = True
        else:
            # For large kernels, these can be expensive operations
            start = time.time()
            # import pdb; pdb.set_trace()
            knl, sb_knl = func_timeout(
                timeout, apply_transformation_list, args=(knl_base, trans_list,))
            dt = time.time() - start
            knl = func_timeout(timeout - dt, lp.preprocess_kernel, args=(knl,))
            insn_ids = tuple(
                [insn.id for insn in knl.default_entrypoint.instructions])
            dt = time.time() - start
            group_sizes, local_sizes = func_timeout(
                timeout - dt, knl.default_entrypoint.get_grid_sizes_for_insn_ids, args=(insn_ids, None,))
            end = time.time()
            transformed = True
            print("Transformation, preprocessing, and obtaining grid sizes required",
                  end - start, "seconds")
            # exit()
    except FunctionTimedOut as e:
        print("Transformation, preprocessing, and obtaining grid sizes timed out")
        transformed = False
        knl = knl_base
        sb_knl = None
        local_sizes = []

    # local_sizes = set()
        # elif trans[0] = "add_prefetch":

        # elif trans[0] == "split_iname" and "inner" in trans[1][0]:
        # Incorrect if the axis is split only once
        # Incorrect if two local axes have the same size
        # local_sizes |= {trans[1][1]}

    # workitems = np.product(list(local_sizes))

    # insn_ids = [insn.id for insn in knl.default_entrypoint.instructions]
    # group_sizes, local_sizes = knl.default_entrypoint.get_grid_sizes_for_insn_ids(insn_ids, None)
    workitems = np.product([entry.max_val() for entry in local_sizes])

    print("WORKITEMS:", workitems)
    # print(local_sizes)
    # print(local_sizes2)
    # for entry in local_sizes2:
    #    print(str(entry.max_val()))
    # exit()
    # assert len(local_sizes) <= 2

    # AMD does something weird with max_work_group_size so using
    # max_work_item_sizes[0] here instead
    max_work_group_size = queue.device.max_work_item_sizes[0]

    temp_dict = {key: val for key, val in knl.default_entrypoint.temporary_variables.items(
    ) if val.address_space == lp.AddressSpace.LOCAL or val.address_space == lp.auto}
    # print(knl)
    #print(temp_dict)
    base_storage_dict = {}

    # This doesn't account for global barriers.
    for temp, tarray in temp_dict.items():
        if tarray.base_storage not in base_storage_dict:
            storage_size = np.product(tarray.shape)*tarray.dtype.dtype.itemsize
            if tarray.base_storage is None:
                # Storage isn't aliased
                base_storage_dict[tarray.name] = storage_size
            else:
                # Storage is aliased
                if tarray.base_storage not in base_storage_dict:
                    base_storage_dict[tarray.base_storage] = storage_size
                elif storage_size > base_storage_dict[tarray.base_storage]:
                    base_storage_dict[tarray.base_storage] = storage_size

    #print("BASE STORAGE DICT")
    #print(base_storage_dict)
    local_memory_used = np.sum(list(base_storage_dict.values()))
    local_memory_avail = queue.device.local_mem_size
    print(
        f"KERNEL USING {local_memory_used} out of {local_memory_avail} bytes of local memory")

    # Could also look at the amount of cache space used and forbid running those that spill

    print("BEGINNING KERNEL EXECUTION")

    if run_single_batch and sb_knl is not None:
        knl = sb_knl

    measured_latency = None
    # Don't allow complete filling of local memory
    if transformed and local_memory_used <= queue.device.local_mem_size and workitems <= max_work_group_size:

        # Should check what the performance difference is between None, subprocess, and thread
        if method is None:
            print("Executing in existing process with no timeout")
            start = time.time()
            # Thread and subprocess don't currently accept the measure_latency parameter
            _, avg_time, measured_latency = test_fn(queue, knl, measure_latency=measure_latency)
            end = time.time()
            wall_clock_time = end - start
        elif method == "subprocess":
            print("Executing test subprocess with timeout of", timeout, "seconds")
            avg_time, measured_latency, wall_clock_time = run_subprocess_with_timeout(queue, knl, test_fn,
                                                                                      timeout=timeout, error_return_time=error_return_time)
        elif method == "thread":
            # Concurrent futures with threads should do the same thing
            try:
                print("Executing test thread with timeout of", timeout, "seconds")
                start = time.time()
                _, avg_time, measured_latency = func_timeout(
                    timeout, test_fn, args=(queue, knl,))
                end = time.time()
                wall_clock_time = end - start
            except Exception as e:
                print(e)
                print("Run failed and threw and exception. Returning error return time.")
                avg_time, wall_clock_time = error_return_time, error_return_time
            # Can probably delete the rest of these.
            except FunctionTimedOut as e:
                print("Execution timed out")
                # Don't run and return return an infinite run time
                avg_time, wall_clock_time = error_return_time, error_return_time
            except cl._cl.RuntimeError as e:
                print(e)
                print("Run failed due to a CL runtime error. Returning error return time.")
                avg_time, wall_clock_time = error_return_time, error_return_time
            except lp.diagnostic.LoopyError as e:
                print("Loopy raised an error. Returning error return time.")
                print(e)
                avg_time, wall_clock_time = error_return_time, error_return_time
            except IndexError as e:
                print("IndexError raised during code generation. Returning error return time.")
                print(e)
                avg_time, wall_clock_time = error_return_time, error_return_time

        else:  # processpool and pebble concurrent processes will break with MPI, use subprocess instead
            avg_time, measured_latency, wall_clock_time = run_concurrent_test_with_timeout(
                queue, knl, test_fn, timeout=timeout, method=method)
    else:
        print("Invalid kernel: Pre-execution timed out, too much local memory was used, or the number of work items exceeded the maximum work group size.")
        # Don't run and return return a large run time
        avg_time, wall_clock_time = error_return_time, error_return_time

    if measured_latency is None or measured_latency == 0:
        measured_latency = device_latency

    print(trans_list)
    print("MAX_FLOP_RATE", max_flop_rate)
    print("COPY LATENCY:", device_latency)
    print("NULL KERNEL LATENCY:", measured_latency)
    print("DEVICE MEMORY BANDWIDTH:", device_memory_bandwidth)

    for trans in trans_list:
        if trans[0] == "batch_einsums":
            batch_size = trans[1][0]
    
    if device_latency is not None and measured_latency is not None:
        measured_latency = min(device_latency, measured_latency)

    if measured_latency is None:
        measured_latency = 0

    data = {"avg_time": avg_time,
            "avg_time_predicted": measured_latency + (avg_time - measured_latency)*(neinsums/batch_size) if (run_single_batch and avg_time != error_return_time) else avg_time,
            "wall_clock_time": wall_clock_time,
            "single_batch": run_single_batch,
            "neinsums": neinsums,
            "error_return_time": error_return_time,
            "timeout": timeout,
            "local_memory_available": local_memory_avail,
            "local_memory_used": local_memory_used,
            "name": knl_base.default_entrypoint.name}

    bw_dict = analyze_knl_bandwidth(
        knl, avg_time, device_latency=measured_latency)
    print("ANALYZED BANDWIDTH")

    bw = bw_dict["observed_bandwidth"]
    data.update(bw_dict.items())

    if device_memory_bandwidth is not None:
        frac_peak_bandwidth = bw / device_memory_bandwidth
        data.update({"frac_peak_bandwidth": frac_peak_bandwidth,
                     "device_memory_bandwidth": device_memory_bandwidth})

    try:
        if timeout is None:
            # Need to use knl rather than knl_base because knl may be a subkernel.
            flop_rate_dict = analyze_flop_rate(
                knl, avg_time, max_flop_rate=max_flop_rate, latency=None)
        else:
            flop_rate_dict = func_timeout(
                timeout, analyze_flop_rate, args=(knl, avg_time, max_flop_rate,))

        flop_rate = flop_rate_dict["observed_flop_rate"]
        data.update(flop_rate_dict.items())

        # if frac_peak_bandwidth  >= bandwidth_cutoff:  # noqa
        #    # Should validate result here
        #    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
        #    return choices

        # This is incorrect for general einsum kernels
        if max_flop_rate is not None:
            # analyze_FLOPS(knl, max_gflops, avg_time)
            frac_peak_flop_rate = flop_rate / max_flop_rate
            data.update({"frac_peak_flop_rate": frac_peak_flop_rate,
                         "max_flop_rate": max_flop_rate})

        # if frac_peak_gflops >= gflops_cutoff:
        #    # Should validate result here
        #    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
        #    return choices

        if max_flop_rate is not None and device_memory_bandwidth is not None:
            roofline_flop_rate = get_knl_device_memory_roofline(knl, max_flop_rate,
                                                                measured_latency, device_memory_bandwidth)
            frac_roofline_flop_rate = flop_rate / roofline_flop_rate

            print("Roofline GFLOP/s:", roofline_flop_rate*1e-9)
            print()

            data.update({"frac_roofline_flop_rate": frac_roofline_flop_rate,
                         "roofline_flop_rate": roofline_flop_rate})

    except FunctionTimedOut as e:
        print("FLOP count timed out. Unable to report FLOP rate.")

    print("ANALYZED FLOP RATE")

    retval = immutabledict({"transformations": trans_list, "data": data})
    print(retval)
    return retval


def exhaustive_search_v2(queue, knl, test_fn, pspace_generator, tlist_generator, time_limit=float("inf"), max_gflops=None,
                         device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):

    # param_list = gen_autotune_list(queue, knl, start_param=start_param)

    # Probably don't need all of these parameters
    # apply_transformations_and_run_test(queue, knl, test_fn, params, max_gflops=None,
    # device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):

    # Should probably obtain device_memory_bandwidth from empirical tests

    # Also fixes the parameters. Maybe that should be a separate function
    # knl = gac.set_memory_layout(knl)

    knl_base = knl.copy()

    params_list = pspace_generator(queue, knl, start_param=start_param)
    # print(knl)
    # print(len(params_list))

    result_list = []
    start = time.time()

    # Iterate over parameter space coordinates
    # If serial run this otherwise, run the parallel autotuner
    # Should probably make separate function for each.
    for params in params_list:
        print(f"Currently testing: {params}")
        """
        trans_list = tlist_generator(params, knl=knl)
        knl = apply_transformation_list(knl_base, trans_list)
        dev_arrays, avg_time = test_fn(queue, knl)

        # Should this return the fraction of peak of should that be calculated in this function?
        gflops, frac_peak_gflops = analyze_FLOPS(knl, avg_time, max_gflops=max_gflops)
        bw = analyze_knl_bandwidth(knl, avg_time)

        if device_memory_bandwidth is not None:  # noqa
            bw = analyze_knl_bandwidth(knl, avg_time)
            frac_peak_GBps = bw / device_memory_bandwidth
            if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                # Should validate result here
                print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                return choices

        # This is incorrect for general einsum kernels
        if max_gflops is not None:
            frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
            if frac_peak_gflops >= gflops_cutoff:
                # Should validate result here
                print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                return choices

        data = None
        if device_memory_bandwidth is not None and max_gflops is not None:
            data = (frac_peak_GBps*device_memory_bandwidth, 
                    frac_peak_gflops*max_gflops, 
                    frac_peak_GBps, 
                    frac_peak_gflops)
        """

        avg_time, trans_list, data = run_single_param_set(
            queue, knl_base, tlist_generator, params, test_fn, max_gflops=max_gflops, device_memory_bandwidth=device_memory_bandwidth)
        result_list.append((avg_time, trans_list, data))
        print(avg_time)
        # result_list.append(data)
        # f.write(str(data) + "\n")

        # if avg_time < avg_time_saved:
        #    avg_time_saved = avg_time
        #    result_saved = choices
        #    result_saved_list = trans_list

        if time.time() - start > time_limit:
            break
            # result_list.sort()
            # print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
            # for entry in result_list:
            #    print(entry)
            # print()

        # return result_saved_list
        # return result_saved

    # print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    # for entry in result_list:
    #    print(entry)
    # print()

    # print("Suggested loop splittings")
    # print(result_saved)
    # print(f"iel: {kio}")
    # print(f"iel_inner: {kii}")
    # print(f"idof: {iio}")
    # print(f"idof_inner: {iii}")
    # print(f"j: {ji}")

    # return result_saved_list
    # return result_saved

    # Could save the highest performing function, but often one wants to see the results
    # over the entire parameter space

    def key_func(result): return result[0]
    sorted_results = sorted(result_list, key=key_func)
    return sorted_results[0]


def exhaustive_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None,
                      device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):

    # Should probably obtain device_memory_bandwidth from empirical tests

    # Imports
    from grudge.grudge_tags import ParameterValue

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size

    avg_time_saved = float("inf")
    result_saved = None

    transform_list = []

    for arg in knl.default_entrypoint.args:
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
                # n_in = n_out # Not true for non-square
            elif IsSepVecOpArray() in arg.tags:
                n_mat, n_out, n_in = arg.shape
            elif IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
            elif IsFaceDOFArray() in arg.tags:
                nfaces, n_elem, n_in = arg.shape
        else:
            if IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize

    # Also fixes the parameters
    # knl = gac.set_memory_layout(knl)

    tested = []

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # k_inner_inner_opt = k_inner_inner_options(start_val=kii_s)
    # kii_s = None
    # j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()

    avg_time_saved = float("inf")
    result_saved = None
    result_saved_list = []

    # Iterate over five search dimensions
    result_list = []
    start = time.time()
    with open("output.txt", "a") as f:
        for kii in k_inner_inner_options(start_val=kii_s):
            # This prevents shared memory from overflowing when running with the face mass kernel
            if knl.default_entrypoint.name == "face_mass":
                n_in_2 = n_in * nfaces
            else:
                n_in_2 = n_in
            for kio in k_inner_outer_options(n_in_2, kii, local_mem_size, fp_bytes=fp_bytes, start_val=kio_s):
                kio_s = None  # Set to None so will form the full set the next time around
                for iii in i_inner_inner_options(n_out, kii,
                                                 max_work_group_size=max_work_group_size, start_val=iii_s):
                    iii_s = None
                    for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                        iio_s = None
                        for ji in j_inner_options(n_in, start_val=ji_s):
                            ji_s = None
                            print((kio, kii, iio, iii, ji))
                            # Transform and run
                            knl = knl_base.copy()
                            knl = lp.split_iname(
                                knl, "iel", kio, outer_tag="g.0", slabs=(0, 1))
                            knl = lp.split_iname(
                                knl, "iel_inner", kii, outer_tag="ilp", inner_tag="l.0", slabs=(0, 1))
                            knl = lp.split_iname(
                                knl, "idof", iio, outer_tag="g.1", slabs=(0, 0))
                            knl = lp.split_iname(
                                knl, "idof_inner", iii, outer_tag="ilp", inner_tag="l.1", slabs=(0, 0))

                            if knl.default_entrypoint.name == "face_mass":
                                knl = lp.add_prefetch(
                                    knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                # knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2") # Should be this but breaks
                            elif knl.default_entrypoint.name == "nodes":
                                knl = lp.add_prefetch(
                                    knl, "nodes", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                knl = lp.tag_array_axes(knl, "vecf", "f,f")
                            elif "resample_by_mat" in knl.default_entrypoint.name:  # Reads are scattered so prefetching is difficult
                                pass
                                # knl = lp.add_prefetch(knl, "ary", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                # knl = lp.tag_array_axes(knl, "vecf", "f,f")
                            else:
                                knl = lp.add_prefetch(
                                    knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                knl = lp.tag_array_axes(knl, "vecf", "f,f")

                            knl = lp.split_iname(
                                knl, "j", ji, outer_tag="for", inner_tag="for")
                            knl = lp.add_inames_for_unused_hw_axes(knl)

                            # Change this to just use the transformation list instead of applying the transformations
                            # directly
                            trans_list = []
                            if "diff" in knl.default_entrypoint.name:
                                trans_list.append(
                                    ["tag_inames", ["imatrix: ilp"]])
                            trans_list.append(["split_iname", ["iel", kio], {
                                              "outer_tag": "g.0", "slabs": (0, 1)}])
                            trans_list.append(["split_iname", ["iel_inner", kii],
                                               {"outer_tag": "ilp", "inner_tag": "l.0", "slabs": (0, 1)}])
                            trans_list.append(["split_iname", ["idof", iio], {
                                              "outer_tag": "g.1", "slabs": (0, 0)}])
                            trans_list.append(["split_iname", ["idof_inner", iii],
                                               {"outer_tag": "ilp", "inner_tag": "l.1", "slabs": (0, 1)}])

                            if knl.default_entrypoint.name == "face_mass":
                                pass
                                # trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
                                #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                                # trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
                            elif knl.default_entrypoint.name == "nodes":
                                trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                                                   {"temporary_name": "vecf", "default_tag": "l.auto"}])
                                trans_list.append(
                                    ["tag_array_axes", ["vecf", "f,f"]])
                            elif "resample_by_mat" in knl.default_entrypoint.name:
                                # Indirection may prevent prefetching
                                pass
                            else:
                                trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                                                   {"temporary_name": "vecf", "default_tag": "l.auto"}])
                                trans_list.append(
                                    ["tag_array_axes", ["vecf", "f,f"]])

                            trans_list.append(["split_iname", ["j", ji], {
                                              "outer_tag": "for", "inner_tag": "for"}])
                            trans_list.append(
                                ["add_inames_for_unused_hw_axes"])

                            print(knl.default_entrypoint.name)
                            print(trans_list)

                            # Execute and analyze the results
                            dev_arrays, avg_time = test_fn(queue, knl)

                            choices = (kio, kii, iio, iii, ji)
                            """
                            if device_memory_bandwidth is not None:  # noqa
                                #frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out,
                                #    n_in, n_elem, max_gflops, device_memory_bandwidth,
                                #    avg_time)
                                bw  = analyze_knl_bandwidth(knl, avg_time)
                                frac_peak_GBps = bw / device_memory_bandwidth
                                result_list.append((frac_peak_GBps, (kio, kii, iio, iii, ji)))
                                if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    pass
                                    #print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                                    #return (kio, kii, iio, iii, ji)
                            """
                            """
                            # TODO: Fix flop calculation
                            if max_gflops is not None and device_memory_bandwidth is not None:  # noqa
                                frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out,
                                    n_in, n_elem, max_gflops, device_memory_bandwidth,
                                    avg_time)
                                if frac_peak_gflops >= gflops_cutoff or frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                                    return (kio, kii, iio, iii, ji)
                            """
                            print(choices)
                            if device_memory_bandwidth is not None:  # noqa
                                bw = analyze_knl_bandwidth(knl, avg_time)
                                frac_peak_GBps = bw / device_memory_bandwidth
                                if frac_peak_GBps >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                                    return choices

                            if max_gflops is not None:
                                frac_peak_gflops = analyze_FLOPS(
                                    knl, max_gflops, avg_time)
                                if frac_peak_gflops >= gflops_cutoff:
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                                    return choices

                            if device_memory_bandwidth is not None and max_gflops is not None:
                                data = (avg_time,
                                        frac_peak_GBps*device_memory_bandwidth,
                                        frac_peak_gflops*max_gflops,
                                        frac_peak_GBps,
                                        frac_peak_gflops,
                                        (kio, kii, iio, iii, ji))
                                result_list.append(data)
                                f.write(str(data) + "\n")

                            if avg_time < avg_time_saved:
                                avg_time_saved = avg_time
                                result_saved = choices
                                result_saved_list = trans_list
                            if time.time() - start > time_limit:
                                result_list.sort()
                                print(
                                    "Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
                                for entry in result_list:
                                    print(entry)
                                print()

                                # return result_saved_list
                                return result_saved

    result_list.sort()

    print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    for entry in result_list:
        print(entry)
    print()

    print("Suggested loop splittings")
    print(result_saved)
    # print(f"iel: {kio}")
    # print(f"iel_inner: {kii}")
    # print(f"idof: {iio}")
    # print(f"idof_inner: {iii}")
    # print(f"j: {ji}")

    return result_saved_list
    # return result_saved


def random_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None,
                  device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95):

    # Imports
    from random import choice
    from grudge.grudge_tags import ParameterValue

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size

    avg_time_saved = float("inf")
    result_saved = None
    result_saved_list = []

    # Get sizes
    for arg in knl.default_entrypoint.args:
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
                # n_in = n_out
            elif IsSepVecOpArray() in arg.tags:
                n_mat, n_out, n_in = arg.shape
            elif IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
            elif IsFaceDOFArray() in arg.tags:
                nfaces, n_elem, n_in = arg.shape
        else:
            if IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize

    # Also fixes the parameters
    knl = gac.set_memory_layout(knl)

    tested = []

    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()
    result_list = []

    start = time.time()
    while (time.time() - start < time_limit):
        # Can be more intelligent by ensuring choices are not run multiple times
        # Maybe could use expressions
        kii = choice(k_inner_inner_opt)
        if knl.default_entrypoint.name == "face_mass":
            kio = choice(k_inner_outer_options(
                n_in*nfaces, kii, local_mem_size, fp_bytes=fp_bytes))
        else:
            kio = choice(k_inner_outer_options(
                n_in, kii, local_mem_size, fp_bytes=fp_bytes))
        iii = choice(i_inner_inner_options(
            n_out, kii, max_work_group_size=max_work_group_size))
        iio = choice(i_inner_outer_options(n_out, iii))
        ji = choice(j_inner_opt)
        choices = (kio, kii, iio, iii, ji)

        if choices not in tested:
            print(choices)
            knl = knl_base.copy()
            if "diff" in knl.default_entrypoint.name:
                knl = lp.tag_inames(knl, "imatrix: ilp")
            knl = lp.split_iname(
                knl, "iel", kio, outer_tag="g.0", slabs=(0, 1))
            knl = lp.split_iname(knl, "iel_inner", kii,
                                 outer_tag="ilp", inner_tag="l.0", slabs=(0, 1))
            knl = lp.split_iname(
                knl, "idof", iio, outer_tag="g.1", slabs=(0, 0))
            knl = lp.split_iname(knl, "idof_inner", iii,
                                 outer_tag="ilp", inner_tag="l.1", slabs=(0, 0))

            if knl.default_entrypoint.name == "face_mass":
                knl = lp.add_prefetch(
                    knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                # Both N1,N0,N2 and N0,N1,N2 both seem to give memory errors..
                # knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2")
            elif knl.default_entrypoint.name == "nodes":
                knl = lp.add_prefetch(
                    knl, "nodes", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "f,f")
            elif "resample_by_mat" in knl.default_entrypoint.name:
                pass
                # Indirection may prevent prefetching
                # knl = lp.add_prefetch(knl, "ary", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                # knl = lp.tag_array_axes(knl, "vecf", "f,f")
            else:
                knl = lp.add_prefetch(
                    knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "f,f")

            knl = lp.split_iname(
                knl, "j", ji, outer_tag="for", inner_tag="for")
            knl = lp.add_inames_for_unused_hw_axes(knl)

            # Change this to just use the transformation list instead of applying the transformations
            # directly
            trans_list = []
            if "diff" in knl.default_entrypoint.name:
                trans_list.append(["tag_inames", ["imatrix: ilp"]])
            trans_list.append(["split_iname", ["iel", kio], {
                              "outer_tag": "g.0", "slabs": (0, 1)}])
            trans_list.append(["split_iname", ["iel_inner", kii],
                               {"outer_tag": "ilp", "inner_tag": "l.0", "slabs": (0, 1)}])
            trans_list.append(["split_iname", ["idof", iio], {
                              "outer_tag": "g.1", "slabs": (0, 0)}])
            trans_list.append(["split_iname", ["idof_inner", iii],
                               {"outer_tag": "ilp", "inner_tag": "l.1", "slabs": (0, 1)}])

            if knl.default_entrypoint.name == "face_mass":
                trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
                                   {"temporary_name": "vecf", "default_tag": "l.auto"}])
                # trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
            elif knl.default_entrypoint.name == "nodes":
                trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                                   {"temporary_name": "vecf", "default_tag": "l.auto"}])
                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
            elif "resample_by_mat" in knl.default_entrypoint.name:
                # Indirection may prevent prefetching
                pass
            else:
                trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                                   {"temporary_name": "vecf", "default_tag": "l.auto"}])
                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

            trans_list.append(["split_iname", ["j", ji], {
                              "outer_tag": "for", "inner_tag": "for"}])
            trans_list.append(["add_inames_for_unused_hw_axes"])

            dev_arrays, avg_time = test_fn(queue, knl)
            tested.append(choices)

            print(choices)
            if device_memory_bandwidth is not None:  # noqa
                bw = analyze_knl_bandwidth(knl, avg_time)
                frac_peak_GBps = bw / device_memory_bandwidth
                # result_list.append((frac_peak_GBps, (kio, kii, iio, iii, ji)))
                if frac_peak_GBps >= bandwidth_cutoff:  # noqa
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                    return choices

            if max_gflops is not None:
                frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
                if frac_peak_gflops >= gflops_cutoff:
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                    return choices

            if device_memory_bandwidth is not None and max_gflops is not None:
                result_list.append((avg_time, frac_peak_GBps*device_memory_bandwidth, frac_peak_gflops*max_gflops,
                                    frac_peak_GBps, frac_peak_gflops, (kio, kii, iio, iii, ji)))

            if avg_time < avg_time_saved:
                avg_time_saved = avg_time
                result_saved = choices
                result_saved_list = trans_list

    print("Time limit exceeded: returning current best result")

    """
    print("Suggested loop splittings")
    print(f"iel: {kio}")
    print(f"iel_inner: {kii}")
    print(f"idof: {iio}")
    print(f"idof_inner: {iii}")
    print(f"j: {ji}")
    """

    result_list.sort()

    print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    # print("Avg time, Frac peak bandwidth, Frac peak GFlops")
    for entry in result_list:
        print(entry)
    print()
    # print(result_list)

    # return result_saved
    return result_saved_list


#def convert(o):
#    if isinstance(o, np.generic):
#        return o.item()
#    raise TypeError


def autotune_and_save(queue, search_fn, tlist_generator, pspace_generator,  hjson_file_str, time_limit=np.inf):
    from hjson import dump
    try:
        avg_time, transformations, data = search_fn(queue, program, generic_test,
                                                    pspace_generator, tlist_generator, time_limit=time_limit)
    except cl._cl.RuntimeError as e:
        print(e)
        print("Profiling is not enabled and the PID does not match any transformation file. Turn on profiling and run again.")

    od = {"transformations": transformations}
    out_file = open(hjson_file_str, "wt")

    hjson.dump(od, out_file, default=convert)
    out_file.close()
    return transformations


def get_transformation_id(device_id):
    hjson_file = open("device_mappings.hjson")
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od[device_id]


if __name__ == "__main__":

    import sys

    unpickle_and_run_test(*sys.argv[1:])

    """
    from .apply_transformations import gen_diff_knl, load_transformations_from_file, apply_transformation_list
    from grudge.execution import diff_prg, elwise_linear_prg, face_mass_prg

    # Test existing optimizations
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Testing code
    device_id = "NVIDIA Titan V"
    tid = get_transformation_id("NVIDIA Titan V")
    fp_format = np.float64
    fp_format_dict = {np.float32: (4, "FP32"), np.float64: (8, "FP64"),
                        np.complex128: (16, "C128")}
    fp_bytes, fp_string = (8, "FP64") if fp_format == np.float64 else (4, "FP32")
    """

    """
    to_test = True
    if to_test:
        n_elem = 2**22#2**15  # 2**21
        pn = 5
        print(len(equidistant_nodes(pn, 3)[1]))
        n_out = len(equidistant_nodes(pn, 3)[1])
        n_in = len(equidistant_nodes(pn, 3)[1])

        #settings = exhaustiveSearch(n_in, n_out, n_elem, 4*12*1024, fp_bytes=fp_bytes,
        #               max_gflops=12288, device_memory_bandwidth=540)
        settings = randomSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=120,
                        fp_format=fp_format, max_gflops=12288//2,
                        device_memory_bandwidth=540)
        #settings = noSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=180,1
        #                       fp_bytes=fp_bytes, max_gflops=12288,
        #                       device_memory_bandwidth=540)
        print("FINAL RESULTS")
        print(settings)
    # Add functionality to write transformations to file
    """
    """
    dim_to_file = {1: "diff_1d_transform.hjson", 
                   2: "diff_2d_transform.hjson",
                   3: "diff_3d_transform.hjson"}

    bandwidths = []
    from os import environ
    for nreg in range(57,58):#range(1, 61):
        environ['CU_JIT_MAX_REGISTERS'] = str(nreg)
        for dim in range(3,4):
            hjson_file = open(dim_to_file[dim])
            #for i in range(2,8):
            pn = 5
            n_out = len(equidistant_nodes(pn, 3)[1])
            n_in = len(equidistant_nodes(pn, 3)[1]) 
            n_elem = 178746 # 2**20
            knl = diff_prg(dim, n_elem, n_out, fp_format) 
            #knl = gen_diff_knl_fortran2(dim, n_elem, n_out, n_in, fp_format=fp_format)
            knl = set_memory_layout(knl)
            knl = lp.set_options(knl, "write_code")
            trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(n_out)])
            knl = apply_transformation_list(knl, trans)
            #print(lp.generate_code_v2(knl).device_code())

            dev_arrays, avg_time = generic_test(queue, knl, nruns=10, warmup=True)
            #dev_arrays, avg_time = runTest(n_elem, n_in, n_out, kio, kii, iio, iii, ji)
            bw = analyze_knl_bandwidth(knl, avg_time)
            bandwidths.append(bw)
            #analyzeResult(n_out, n_in, n_elem, 12288//2, 540, avg_time, fp_bytes=fp_bytes)
            print(avg_time)
            #verifyResult(*dev_arrays)
    
    print(knl)
    for i, entry in enumerate(bandwidths):
        print(f"{i}, {entry}")
    #print(bandwidths)
    """
    # testBandwidth()
    # exit()
    """
    # Test elwise linear
    pn = 4
    n_out = len(equidistant_nodes(pn,3)[1])
    n_in = n_out
    n_elem = 178746
    fp_format = np.float64
    fp_string = "FP64" if fp_format == np.float64 else "FP32" 
    knl = elwise_linear_prg(n_elem, n_out, fp_format)
    #knl = gen_elwise_linear_knl(n_elem, n_in, n_out, fp_format)

    hjson_file = open("elwise_linear_transform.hjson")
    trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(n_out)])

    knl = set_memory_layout(knl)
    knl = apply_transformation_list(knl, trans)
    #print(knl)
    _, avg_time = generic_test(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    print(avg_time)
    analyze_knl_bandwidth(knl, avg_time)
    """
    """
    # Test face_mass            
    pn = 3
    nvol_nodes = len(equidistant_nodes(pn,3)[1])
    nface_nodes = 10
    #nelements = 2**22
    nelements = 178746
    nfaces = 4
    fp_format = np.float64
    fp_string = "FP64" if fp_format == np.float64 else "FP32" 

    knl = face_mass_prg(178746, 4, 20, 20, np.float64)
    knl = set_memory_layout(knl)
    #knl = gen_face_mass_knl(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    #knl = gen_face_mass_knl_merged(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    # Need to load these from file
    #hjson_file = open("elwise_linear_transform.hjson")
    #trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(pn)])
    #knl = apply_transformation_list(knl, trans)
    print(knl)
    _, avg_time = test_face_mass(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    #_, avg_time = test_face_mass_merged(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    print(avg_time)
    analyze_knl_bandwidth(knl, avg_time)
    """

    # Test order=4 copy
    """
    knl = lp.make_copy_kernel("f,f", old_dim_tags="f,f")
    knl = lp.add_dtypes(knl, {"input": np.float64, "output": np.float64})
    knl = lp.fix_parameters(knl, {"n0": 178746, "n1": 35})  
    knl = lp.split_iname(knl, "i0", 48, outer_tag="g.0")
    knl = lp.split_iname(knl, "i0_inner", 16, outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_iname(knl, "i1", 35, outer_tag="g.1", inner_tag="l.1")
    for arg in knl.default_entrypoint.args:
        if arg.name == "input":
            arg.tags = IsDOFArray()
            arg.shape = (178746, 35)
        if arg.name == "output":
            arg.tags = IsDOFArray()
            arg.is_output = True 
            arg.shape = (178746, 35)

    print(knl)
    _, avg_time = generic_test(queue, knl)
    analyze_knl_bandwidth(knl, avg_time)
    #knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 1024, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0") 
    #knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1)) 
    #knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))
    """

    # """
    # Test autotuner
    # knl = diff_prg(3, 1000000, 3, np.float64)
    # print(knl)
    # print(knl.default_entrypoint.domains)
    # print(knl.default_entrypoint.instructions)
    # exit()
    # knl = diff_prg(3, 196608, 10, np.float64)
    # knl = elwise_linear_prg(24576, 120, np.float64)
    # dofs = 84
    # knl = elwise_linear_prg(1000000, 3*dofs, np.float64, nnodes_in=dofs)
    # start_param = (24, 4, 126, 9, 28)#(96, 32, 60, 2, 5)
    # start_param = None
    # Figure out the actual dimensions
    # knl = face_mass_prg(178746, 4, 20, 20, np.float64)

    # Spock
    # result = exhaustive_search(queue, knl, generic_test, time_limit=np.inf, max_gflops=11540, device_memory_bandwidth=1047, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
    # pspace_generator = gen_autotune_list(queue, knl)
    # print(len(result))

    # Titan V
    # result = exhaustive_search(queue, knl, generic_test, time_limit=np.inf, max_gflops=6144, device_memory_bandwidth=580, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
    # print(result)
    # pspace_generator = gen_autotune_list
    # tlist_generator = mxm_trans_list_generator
    # result = exhaustive_search_v2(queue, knl, generic_test, pspace_generator, tlist_generator, time_limit=np.inf, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)

    # result = exhaustive_search_v2(queue, knl, generic_test, pspace_generator, tlist_generator, time_limit=np.inf, max_gflops=6144, device_memory_bandwidth=580, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
