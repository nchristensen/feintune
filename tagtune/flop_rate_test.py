import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array as cla

def generate_flop_instructions(nflops):
    one_flop_instr = "beta = A[i] + alpha\n"
    two_flop_instr = "beta = beta*A[i] + alpha\n"

    half_flops = nflops//2

    if nflops % 2 == 1:
        instr = one_flop_instr + two_flop_instr*half_flops
    else:
        instr = two_flop_instr*half_flops

    print(instr)
    return instr

def generate_test_knl(nflops, nelem, ntrials=100):

    one_flop_instr = "beta = A[i] + alpha {id_prefix=flop,dep=*}" if nflops % 2 == 1 else ""
        
    nfmadd = nflops // 2
    knl = lp.make_kernel(
        "{[j,k,i]: 0<=k<nfmadd and 0<=j<ntrials and 0<=i<nelem}",
        f"""
        for j
            <> alpha = 2.0
            <> beta = 1.0
            {one_flop_instr}
            for k
                beta = A[i] + alpha {{id_prefix=flop,dep=*}}
            end
            A[i] = -beta {{dep=flop*}}
        end
        """,
        assumptions="nfmadd >= 0 and ntrials>=0 and nelem>=0",
        fixed_parameters={"nfmadd": nfmadd, "ntrials": ntrials, "nelem": nelem}
    )

    knl = lp.tag_inames(knl, [("k", "unr",)])
    return knl


if __name__ == "__main__":

   
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    max_size_bytes = queue.device.max_mem_alloc_size
    dtype = np.float64
    max_size_dtype = max_size_bytes // dtype().itemsize

    A = cla.zeros(queue, (max_size_dtype,), dtype) + 1

    flops_per_work_item = 1024
    ntrials = 500
    knl = generate_test_knl(flops_per_work_item, A.shape[0], ntrials=ntrials)
    knl = lp.split_iname(knl, "i", 1024, outer_tag="g.0", inner_tag="l.0")
    knl = lp.add_inames_for_unused_hw_axes(knl)

    evt, result = knl(queue, A=A)
    evt.wait()    

    dt = (evt.profile.end - evt.profile.start) / 1e9
    from run_tests import analyze_FLOPS
    analyze_FLOPS(knl, dt)

    tot_flops = flops_per_work_item*max_size_dtype*ntrials
    gflop_rate = tot_flops/dt/1e9
    print(gflop_rate)

    #generate_flop_instructions(10)
