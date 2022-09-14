import pyopencl as cl
import numpy as np
import pyopencl.array as clarray
import pyopencl.clrandom as clrandom
from pyopencl.tools import ImmediateAllocator

context = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

max_shape = 2**31
shape = 10
allocator = ImmediateAllocator(queue)

d_in_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_WRITE_ONLY, size=max_shape)
d_out_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_NO_ACCESS, size=max_shape)
h_in_buf = np.random.rand(2**29)#np.float32(np.random.rand(max_shape//4))
h_in_buf = np.float32(h_in_buf)

evt = cl.enqueue_copy(queue, d_in_buf, h_in_buf)
evt.wait()

niter=1000
dt = 0

len_list = []
word_count = 1
word_size = 4

while word_count < 2**29:
    len_list.append(word_count)
    word_count = int(np.ceil(word_count*1.5))
for i in range(29):
    len_list.append(2**i)
len_list = sorted(list(set(len_list)))

print(len_list)

for word_count in len_list:
    events = []
    byte_count = word_size*word_count
    # Warmup
    for i in range(2):
        evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
    for i in range(niter):
        evt = cl.enqueue_copy(queue, d_out_buf, d_in_buf, byte_count=byte_count)
        events.append(evt)

    cl.wait_for_events(events)
    for evt in events:
        dt += evt.profile.end - evt.profile.start
    dt  = dt / 1e9

    dt = dt/niter
    nbytes_transferred = 2*byte_count
    print("{} {} {}".format(nbytes_transferred, dt, nbytes_transferred/dt/1e9))

