import pyopencl as cl
import numpy as np
import pyopencl.array as clarray
import pyopencl.clrandom as clrandom
from pyopencl.tools import ImmediateAllocator
from dataclasses import dataclass

# Will get queues for each device with the same name
# since different CL implementations may have different
# performance.
def get_queues_like(queue):
    queues = []
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.name == queue.device.name:
                context = cl.Context(devices=[device])
                queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
                queues.append(queue)
    return queues

@dataclass
class BandwidthTestResult():

    device: str
    tavg: float
    tmin: float
    tmax: float
    bytes_transferred: int
    #nbytes_read: int = None
    #nbytes_written: int = None

    @property
    def avg_bandwidth(self):
        return self.bytes_transferred/self.tavg

    @property
    def max_bandwidth(self):
        return self.bytes_transferred/self.tmin

    @property
    def min_bandwidth(self):
        return self.bytes_transferred/self.tmax

    
def enqueue_copy_bandwidth_test_with_queues_like(queue, dtype=None, fill_on_device=True, max_used_bytes=None):

    queues = get_queues_like(queue)

    return tuple([enqueue_copy_bandwidth_test(q, dtype=dtype,
                    fill_on_device=fill_on_device,
                    max_used_bytes=max_used_bytes) for q in queues])


def get_buffers(queue, dtype_in, n_dtype_in, dtype_out=None, n_dtype_out=None, fill_on_device=True):

    if n_dtype_out is None:
        n_dtype_out = n_dtype_in
    if dtype_out is None:
        dtype_out = dtype_in

    n_bytes_in = n_dtype_in*dtype_in().itemsize
    n_bytes_out = n_dtype_out*dtype_out().itemsize
    context = queue.context

    d_out_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_NO_ACCESS, size=n_bytes_out)

    if fill_on_device: # Requires making a READ_WRITE buffer instead of a READ_ONLY buffer
        if dtype_in in {np.float64, np.float32, np.int32, np.int64}:
            allocator = ImmediateAllocator(queue)
            d_in_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS, size=n_bytes_in)
            d_in_buf_arr = cl.array.Array(queue, (n_dtype_in,), dtype_in, allocator=allocator, data=d_in_buf)
            clrandom.fill_rand(d_in_buf_arr, queue=queue)
        else:
            raise ValueError(f"Cannot fill array with {dtype} on the device")
    else:

        from psutil import virtual_memory
        
        if np.issubdtype(dtype, np.integer):
            if virtual_memory().available < n_bytes_in:
                raise ValueError("Not enough host memory to fill the buffer from the host")

            max_val = np.iinfo(dtype).max
            min_val = np.iinfo(dtype).min
            h_in_buf = np.random.randint(min_val, high=max_val + 1, size=max_shape_dtype, dtype=dtype_in)
        elif np.issubdtype(dtype, np.float):
            # The host array is formed as a float64 before being copied and converted
            if virtual_memory().available < n_dtype_in*(np.float64().itemsize + dtype_in().itemsize):
                raise ValueError("Not enough host memory to fill the buffer from the host")
            h_in_buf = np.random.rand(n_dtype_in).astype(dtype_in)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        d_in_buf = cl.Buffer(context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.COPY_HOST_PTR,
            size=n_bytes_in, hostbuf=h_in_buf)

        
        #TODO: Copy small chunks at a time if the array size is large.
        # Is this actually needed?
 
        #d_in_buf = cl.Buffer(context,
        #    cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
        #    size=max_shape_bytes, hostbuf=h_in_buf)
        #for some number of chunks           
        #   h_in_buf = ...
        #   evt = cl.enqueue_copy(queue, d_in_buf, h_in_buf) # With offsets

    return d_in_buf, d_out_buf

def get_word_counts(max_shape_dtype):
    word_count_list = []

    word_count = 1
    # Get some non-multiples of two
    while word_count <= max_shape_dtype:
        word_count_list.append(int(np.floor(word_count)))
        word_count = word_count*1.5
    # Get multiples of two
    for i in range(0,int(np.floor(np.log2(max_shape_dtype)) + 1)):
        word_count_list.append(2**i)
    word_count_list = sorted(list(set(word_count_list)))
    return word_count_list


def loopy_bandwidth_test(queue, dtype=None, n_in=None, n_out=None,
                        fill_on_device=True, ntrials=1000):

    #knl = lp.make_copy_kernel("c,c", old_dim_tags="c,c")
    n = max(n_in, n_out)
    knl = lp.make_knl(
        "{ [i]: 0<=i<n}",
        """
        out[i % n_out] = in[i % n_in]
        """,
        assumptions="n>=0",
        
    )
    knl = lp.add_dtypes(knl, {"input": fp_format, "output": fp_format})
    knl = knl.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
    n0 = 2
    #knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "i1", 256, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0") 
    #knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1)) 
    #knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))

    fp_bytes = 8 if fp_format == np.float64 else 4

    # This assumes fp32
    len_list = []
    float_count = 1
    max_floats = 2**28
    while float_count <= max_floats:
        len_list.append(float_count)
        float_count = int(np.ceil(float_count*1.5))
    for i in range(29):
        len_list.append(2**i)
    len_list = sorted(list(set(len_list)))

    #data = np.random.randint(-127, 128, (1,max_bytes), dtype=np.int8)
    #inpt = cl.array.to_device(queue, data, allocator=mem_pool)
    from pyopencl.array import sum as clsum

    from pyopencl.tools import ImmediateAllocator, MemoryPool
    allocator = ImmediateAllocator(queue)
    mem_pool = MemoryPool(allocator) 


    print(len_list)

    for n in len_list:
    #for i in range(29):

        #n = 2**i
        kern = lp.fix_parameters(knl, n0=n0, n1=n)
        #data = np.random.randint(-127, 128, (1,n), dtype=np.int8)
        #inpt = cl.array.to_device(queue, data, allocator=mem_pool)
        inpt = cl.clrandom.rand(queue, (n0, n), dtype=fp_format)
        outpt = cl.array.Array(queue, (n0, n), dtype=fp_format, allocator=mem_pool)
     
        #kern = lp.set_options(kern, "write_code")  # Output code before editing it

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
        #queue.finish()
        dt = dt / nruns / 1e9

        nbytes_transferred = 2*fp_bytes*n*n0
        bandwidth = nbytes_transferred / dt / 1e9
        print("{} {}".format(nbytes_transferred, bandwidth))

        #print((inpt - outpt)) 
        diff = (inpt - outpt)
        if  clsum(inpt - outpt) != 0:
            print("INCORRECT COPY")




def enqueue_copy_bandwidth_test(queue, dtype=None, fill_on_device=True, max_used_bytes=None, ntrials=1000):

    if dtype is None:
        dtype = np.int32 if fill_on_device else np.int8


    if max_used_bytes is None:
        max_shape_bytes = queue.device.max_mem_alloc_size
    else:
        max_shape_bytes = max_used_bytes // 2

    word_size = dtype().itemsize
    max_shape_dtype = max_shape_bytes // word_size
    # Redefine max_shape_bytes in case there is a remainder in the division
    max_shape_bytes = max_shape_dtype*word_size
    max_used_bytes = 2*max_shape_bytes

    if max_shape_bytes > queue.device.max_mem_alloc_size:
        raise ValueError("max_shape_bytes is larger than can be allocated")

    d_in_buf, d_out_buf = get_buffers(queue, dtype, max_shape_dtype, fill_on_device=fill_on_device)

    word_count_list = get_word_counts(max_shape_dtype)
    results_list = []
    
    for word_count in word_count_list:
        dt_max = 0
        dt_min = np.inf
        dt_avg = 0

        events = []
        byte_count = word_size*word_count

        # Warmup
        for i in range(5):
            evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
        for i in range(ntrials):
            evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            dt = evt.profile.end - evt.profile.start
            dt_avg += dt
            if dt > dt_max:
                dt_max = dt
            if dt < dt_min:
                dt_min = dt

        # Convert to seconds
        dt_avg  = dt_avg / ntrials / 1e9
        dt_max = dt_max / 1e9
        dt_min = dt_min / 1e9

        # Calculate bandwidth in GBps
        nbytes_transferred = 2*byte_count
        avg_bw = nbytes_transferred/dt_avg/1e9
        max_bw = nbytes_transferred/dt_min/1e9
        min_bw = nbytes_transferred/dt_max/1e9

        result = BandwidthTestResult(str(queue.device), dt_avg, dt_min, dt_max, nbytes_transferred)
        results_list.append(result)

        print(f"{nbytes_transferred} {dt_avg} {dt_min} {dt_max} {avg_bw} {max_bw} {min_bw}")

    return tuple(results_list)

# Returns latency in seconds and inverse bandwidth in seconds per byte
def get_alpha_beta_model(results_list, total_least_squares=False):

    # Could take the latency to be the lowest time ever seen,
    # but that might be limited by the precision of the event timer

    if total_least_squares:
        M = np.array([(1, result.bytes_transferred, result.tmin) for result in results_list])
        U, S, VT = np.linalg.svd(M)
        coeffs = ((-1/VT[-1,-1])*VT[-1,:-1]).flatten()
    else:
        M = np.array([(1, result.bytes_transferred, result.tmin) for result in results_list])
        coeffs = np.linalg.lstsq(M[:,:2], M[:,2], rcond=None)[0]

    return (coeffs[0], coeffs[1],)

def plot_bandwidth(results_list):
    import matplotlib.pyplot as plt

    latency, inv_bandwidth = get_alpha_beta_model(results_list)
    print("LATENCY:", latency, "BANDWIDTH:", 1/inv_bandwidth/1e9)
    M = np.array([(result.bytes_transferred, result.max_bandwidth) for result in results_list])

    best_fit_bandwidth = M[:,0]/(latency + M[:,0]*inv_bandwidth)/1e9
    
    fig = plt.figure()
    plt.semilogx(M[:,0], M[:,1]/1e9)
    plt.semilogx(M[:,0], best_fit_bandwidth)
    plt.xlabel("Bytes read + bytes written")
    plt.ylabel("Bandwidth (GBps)")
    plt.show()

if __name__ == "__main__":
    
    context = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)


    #results_list = enqueue_copy_bandwidth_test(queue, dtype=None, fill_on_device=True, max_used_bytes=None)

    #get_alpha_beta_model(results_list)
    #plot_bandwidth(results_list)

    results_list_list = enqueue_copy_bandwidth_test_with_queues_like(queue, max_used_bytes=None)
    
    key = lambda result: result.tmin
    combined_list = [sorted(tup, key=key)[0] for tup in zip(*results_list_list)]

    for results_list in results_list_list:
        coeffs = get_alpha_beta_model(results_list)
        print(coeffs)

    plot_bandwidth(combined_list)

