import loopy as lp
import numpy as np
from decouple_domain import decouple_domain


t_unit = lp.make_kernel(
    "{[i]: 0<=i<100}",
    """
    x[i] = i {id=batch1}
    y[j] = 2*i {id=batch2}
    """,
    name="foo",
)


def gen_diff_knl(n_elem, n_in, n_out, arch="AMD_GPU", target=lp.OpenCLTarget):
    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        #result[k,i] = sum(j, mat[i, j] * vec[k, j])   
        result[i,k] = sum(j, mat[i, j] * vec[j, k]) #Assume correct memory layout  
        """,
        kernel_data=[
            lp.GlobalArg("result", np.float32, shape=(
                n_out, n_elem), order="C"),
            lp.ConstantArg("mat", np.float32, shape=(n_in, n_out), order="C"),
            lp.GlobalArg("vec", np.float32, shape=(n_out, n_elem), order="C")
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        default_offset=None,
        name="diff"
        # target=target
    )

    if arch == "NVIDIA_GPU":
        wkgpsz = [32*8, 32//8]
        # wkgpsz = [32, 32]
    elif arch == "AMD_GPU":
        wkgpsz = [32, 32]
    else:
        wkgpsz = [32, 32]
        # wkgpsz = 8096

    if n_elem < wkgpsz[0]:
        wkgpsz[0] = n_elem
    if n_out < wkgpsz[1]:
        wkgpsz[1] = n_out

    knl = lp.fix_parameters(knl, nelements=n_elem,
                            ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)
    # knl = lp.add_dtypes(knl, {"mat": np.float32, "vec": np.float32})

    # Basic version
    # knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
    # knl = lp.tag_inames(knl, dict(k="g.0"))

    if arch == "CPU":  # In principle the buffered data can be in global memory
        # Perhaps try chunking (into num processor sized chunks), then splitting
        slabs = (0, 0) if n_elem % wkgpsz == 0 else (0, 1)
        knl = lp.split_iname(knl, "k", wkgpsz, outer_tag="g.0",
                             inner_tag="l.0", slabs=slabs)
        # knl = lp.tag_inames(knl, [("j", "unr"), ("i", "l.1")]) #Results in terrible performance
        knl = lp.tag_inames(knl, [("j", "unr")])

        knl = lp.add_prefetch(knl, "vec", "j,k_inner",
                              temporary_name="vecf", default_tag="l.auto")
        # Fix this so need not assign before using
        # knl = lp.buffer_array(knl, "result", buffer_inames="k_inner,i",
        #    init_expression="0", default_tag="l.auto")
        # Transpose while prefetching
        knl = lp.tag_array_axes(knl, "vecf", "N0,N1")
        # knl = lp.tag_array_axes(knl, "vecf,result_buf", "N0,N1") #Transpose while prefetching

    if arch != "CPU":
        # ktag = "l.0"
        # itag = "l.1"

        slabs0 = (0, 0) if n_elem % wkgpsz[0] == 0 else (0, 1)
        slabs1 = (0, 0) if n_elem % wkgpsz[0]*2 == 0 else (0, 1)
        knl = lp.split_iname(
            knl, "k", wkgpsz[0], outer_tag="g.0", inner_tag="l.0", slabs=slabs0)
        # knl = lp.split_iname(knl, "k", wkgpsz[0]*2, outer_tag="g.0", slabs=slabs1)
        # knl = lp.split_iname(knl, "k_inner", wkgpsz[0], outer_tag="l.0", slabs=slabs0)

        # slabs1 = (0,0) if n_out % wkgpsz[1] == 0 else (0,1)
        # knl = lp.split_iname(knl, "i", wkgpsz[1], outer_tag="g.1",inner_tag="l.1", slabs=slabs1)

        # knl = lp.tag_inames(knl, [("i", itag)])
        # knl = lp.tag_inames(knl, [("j", "unr")]) # Makes no difference on Nvidia GPUs

        # knl = lp.tag_inames(knl, [("j", "unr")]) # Makes no difference on Nvidia GPUs

        knl = lp.add_prefetch(knl, "vec", "j,k_inner",
                              temporary_name="vecf", default_tag="l.auto")
        # knl = lp.add_prefetch(knl, "vec", "j,k_inner_outer", temporary_name="vecf", default_tag="l.auto")
        # knl = lp.add_prefetch(knl, "mat", "i_inner,j", temporary_name="matf", default_tag="l.auto")
        # Fix this so need not assign before using
        # knl = lp.buffer_array(knl, "result", buffer_inames="k_inner,i_inner",
        #    init_expression="0", default_tag="l.auto")

        # knl = lp.tag_array_axes(knl, "matf", "N0,N1") #Transpose while prefetching
        # knl = lp.tag_array_axes(knl, "vecf", "N0,N1") #Transpose while prefetching

        # knl = lp.tag_array_axes(knl, "vecf,result_buf", "N0,N1") #Transpose while prefetching

        # knl = lp.change_arg_to_image(knl, "vec") #Not implemented in CUDA
        # align_bytes=32
        # pad_mult = lp.find_padding_multiple(knl, "mat", 0, align_bytes)
        # knl = lp.split_array_dim(knl, ("mat", 0), pad_mult)
        # knl = lp.add_padding(knl, "matf", 0, align_bytes)

    for entry in knl.default_entrypoint.inames:
        knl = decouple_domain(knl.default_entrypoint, [entry], frozenset())

    code = lp.generate_code_v2(knl).device_code()
    print(code)

    return knl


knl = gen_diff_knl(100000, 35, 35)
print(knl)
