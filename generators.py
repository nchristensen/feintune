import numpy as np
import loopy as lp
#from frozendict import frozendict
from meshmode.transform_metadata import FirstAxisIsElementsTag
from grudge_tags import (IsDOFArray, IsSepVecDOFArray,
    IsOpArray, IsSepVecOpArray, IsFaceDOFArray, IsFaceMassOpArray,
    IsVecDOFArray, IsVecOpArray, IsFourAxisDOFArray)

def k_inner_inner_options(start_val=None):
    #options = [8, 16, 4, 32]
    #options = [64, 32, 16, 8]
    options = [32, 16, 8]
    #options = [32, 16]
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options


def k_inner_outer_options(n_in, k_inner_inner, sm_size,
                            fp_bytes=8, start_val=None, nelem=None):
    # Possibilities limited by size of local memory
    # Use sm_size - 1 because CUDA errors when all of local memory is used
    # Assumes a single DOF array. Additional pruning probably required
    # Assumes we prefetch all of the dofs in a strip of elements. This does not need to be the case
    # though. We could prefetch chunks (of length equal to the j_inner loop?) at a time.
    #options = np.arange(1, ((sm_size - 1) // (fp_bytes*k_inner_inner*n_in)) + 1)

    options = np.arange(1, (sm_size // (fp_bytes*k_inner_inner*n_in)) + 1)


    #Arbitrarily limit to at max 6 inline to limit search space
    ilp_limit = min(nelem // k_inner_inner, 6) if nelem is not None else 6
    options = list(k_inner_inner*options[options <= ilp_limit])

    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options

def i_inner_inner_options(n_out, k_inner_inner, max_work_group_size=1024, start_val=None):
    factors = np.arange(1, n_out+1)[(n_out % np.arange(1, n_out+1)) == 0]
    # Ensure total number of workitems is less than maximum
    usable_factors = factors[factors*k_inner_inner <= max_work_group_size]
    options = sorted(usable_factors, reverse=True)
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options

def i_inner_outer_options(n_out, i_inner_inner, start_val=None):
    # Select a number of inline blocks such that n_out % outer*inner == 0
    # Bumping up the start of the range could reduce autotune time, but an empty
    # autotune set might be returned if i < start value
    
    # Loopy confused about the number of dimensions when 
    # i_outer, i_inner_outer, and i_inner_inner are all 1
    #inline = np.array([1]) if n_out == 1 else np.arange(1, (n_out // i_inner_inner) + 1)

    inline = np.arange(1, max(1,(n_out // i_inner_inner)) + 1)
    options = list(i_inner_inner*inline[n_out % (inline*i_inner_inner) == 0])
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options


def j_inner_options(n_in, start_val=None):

    start = 1
    factors = list(np.arange(start, n_in + 1)[(n_in % np.arange(start, n_in + 1)) == 0])
    #factors = list(np.arange(1, n_in + 1)[(n_in % np.arange(1, n_in + 1)) == 0])
    # Should this be limited by the number of registers
    start_ind = 0 if start_val is None else factors.index(start_val)
    factors = factors[start_ind:]
    return factors

# Creates a list containing tuples of search space parameters.
# Will need to create separate ones of this for each einsum kernel
def gen_autotune_list(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    
    nfaces = 1

    n_in = None
    print(knl.default_entrypoint.name)
    ndof_arrays = 0
    for arg in knl.default_entrypoint.args:
        print(arg.name)
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
                ndof_arrays += 1
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
    ndof_arrays = max(ndof_arrays, 1)
    if n_in is None:
        n_in = n_out

    n_in = n_in * nfaces #Prevents shared memory from overflowing in face mass kernel   

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Should come up with a way to set the effective local memory size. It depends on the number of
        # arrays actually prefetched.
        for kio in k_inner_outer_options(n_in*nfaces, kii, local_mem_size // ndof_arrays, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    # Kernel does not reach here.
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


# Should separate this so don't need to supply knl
def mxm_trans_list_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]


    #if "diff" in knl.default_entrypoint.name:
    #    trans_list.append(["tag_inames", ["imatrix: ilp"]])

    trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["iel_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["idof_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])

    if knl.default_entrypoint.name == "face_mass":
        pass
        #trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
        #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
        #trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
    #elif knl.default_entrypoint.name == "nodes":
    elif knl.default_entrypoint.name == "lp_nodes":
        trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
    elif "resample_by_mat" in knl.default_entrypoint.name:
        # Indirection may prevent prefetching
        pass
    else:
        trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list


def grudge_elementwise_sum_knl_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]

    trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["iel_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["idof_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    #trans_list.append(["add_prefetch", ["operand", "iel_inner_outer,iel_inner_inner"],
    #    {"temporary_name":"operandf", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["operandf", "f,f"]])

    # Realistically, splitting the j loop probably is not necessary for this.
    trans_list.append(["split_iname", ["jdof", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def grudge_elementwise_sum_knl_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            n_in = n_out
            fp_bytes = arg.dtype.dtype.itemsize

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions. Could reduce this to 4 if ignore j-loop.
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both jac and vec are prefetched so the available local_memory per prefetched array is halved
        for kio in k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


def einsum3to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    if 0 not in params: # If there is a zero length dimension then don't transform
        knl = kwargs["knl"]

        if kio != kii:
            trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["e_inner", kii], 
                {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
            prefetch_str = "j,e_inner_outer,e_inner_inner"
        else:
            trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "inner_tag": "l.0", "slabs":(0,0)}])
            prefetch_str = "j,e_inner"    
        if iio != iii:
            trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
            trans_list.append(["split_iname", ["i_inner", iii], 
                {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
        else:
            trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "inner_tag": "l.1", "slabs":(0,0)}])
        # Should the i loop have (0,1) slabs for both?

        for arg in knl.default_entrypoint.args:

            if "vec" == arg.name:
                trans_list.append(["add_prefetch", ["vec", prefetch_str],
                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
            elif "jac" == arg.name:
                trans_list.append(["add_prefetch", ["jac", prefetch_str],
                    {"temporary_name":"jacf", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["jacf", "f,f"]])
            elif "arg2" == arg.name and IsDOFArray() in arg.tags:
                trans_list.append(["add_prefetch", ["arg2", prefetch_str],
                    {"temporary_name":"arg2f", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["arg2f", "f,f"]])
            elif "arg1" == arg.name and IsDOFArray() in arg.tags:
                trans_list.append(["add_prefetch", ["arg1", prefetch_str],
                    {"temporary_name":"arg1f", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])
            elif "arg0" == arg.name and IsDOFArray() in arg.tags:
                arg0_prefetch_str = "i_inner," if iio == iii else "i_inner_outer,i_inner_inner,"
                arg0_prefetch_str += "e_inner" if kio == kii else "e_inner_outer,e_inner_inner"
                trans_list.append(["add_prefetch",
                    ["arg0", arg0_prefetch_str],
                    {"temporary_name":"arg0f", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["arg0f", "f,f"]])

        trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])

    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum3to2_kernel_tlist_generator_v2(queue, knl, **kwargs):

    from __init__ import get_einsum_types, get_einsums
    # Create the list of parameter values to try

    # Does not account for local memory usage due to register spilling
    # Need to reserve some amount for that
    # Count number of constants and private memory objects 
    # and subtract that from available local memory?
    # Would need to multiply that number by the number of concurrent threads, including ilp
    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_item_sizes[0]
    #max_work_group_size = queue.device.max_work_group_size

    # Find read dof arrays. These are the ones that will be prefetched
    read_deps = frozenset()
    write_deps = frozenset()
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            read_deps |= instr.read_dependency_names()
            write_deps |= instr.write_dependency_names()

    dof_arrays = []
    face_dof_arrays = []
    arg_dict = dict([(arg.name, arg) for arg in knl.default_entrypoint.args])
    arg_dict.update(knl.default_entrypoint.temporary_variables)

    print(write_deps)
    print(read_deps)
    n_elem = None
    nx = None
    nr = None
    nf = None
    sizes = frozenset()
    n_dof_arrays = 0
    for arg in list(arg_dict.values()):
        if arg.name in write_deps and len(arg.shape) == 2:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
            break
        elif arg.name in write_deps and len(arg.shape) == 3:
            nx, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
            break
    for arg in list(arg_dict.values()):
        if len(arg.shape) == 2 and arg.name in read_deps and arg.shape[0] == n_elem:
            n_in = arg.shape[1]
            dof_arrays.append(arg.name)
            n_dof_arrays += 1
        elif len(arg.shape) == 3 and arg.name in read_deps and arg.shape[-1] == n_elem:
            _, nr, _ = arg.shape
        elif len(arg.shape) == 3 and arg.name in read_deps and arg.shape[1] == n_elem:
            nf, _, n_in = arg.shape
            n_dof_arrays += nf
            face_dof_arrays.append(arg.name)
            

    # TODO: Enable prefetching of ndim x n_element array. Need to adjust
    # how available local memory is calculated
    # Or just don't prefetch and rely on caching to compensate

    #print(n_elem, n_out, n_in)

    """
    if FirstAxisIsElementsTag() in arg.tags and len(arg.shape) == 2:
        print("HERE")
        dof_arrays.append(arg.name)
        print(arg.name, read_deps)
        if arg.name in read_deps:
            print("HERE2")
            n_elem, n_in = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
    elif len(arg.shape) == 2: # Not super robust
        n_out, n_in = arg.shape
    """

    read_dof_arrays = read_deps & frozenset(dof_arrays)
    #n_dof_arrays = len(read_dof_arrays)

    start_param = (None, None, None, None, None)
    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    
    neinsums = len(get_einsums(knl))
    batch_size_list = [(batch_size, int(np.ceil(neinsums/batch_size))) for batch_size in range(1,neinsums + 1)]
    nbatches_dict = {}

    # Do in reverse so we wind up with the smallest batch size that requires nbatches
    for nbatches, batch_size in reversed(batch_size_list):
        nbatches_dict[nbatches] = batch_size
    batch_size_list = sorted(nbatches_dict.values())

    #batch_size_list = [sorted(nbatches_dict.values())[0]]

    # Very small batch sizes tend to not run because duplicate_inames increases in cost quadratically with the
    # number of inames
    for batch_size in list(reversed(batch_size_list)):#range(3,4):#range(1, neinsums + 1):

        if batch_size >= neinsums:
            batch_size = 0

        if n_elem*n_out <= 1024:
            choices = (batch_size, n_elem, n_elem, n_out, n_out, n_in)
            parameter_list.append(choices)
        else:

            for kii in k_inner_inner_options(start_val=kii_s):
                # Might be easier to generate all of the entries and then prune those
                # entries that use too much local memory.
                # Could also estimate amount of cache needed and prune those that spill.
                # Also should look at the maximum number of dof arrays per batch and 
                # allocate those arrays once and reuse them in every batch
                # is naming them the same sufficient? This current implementation
                # assumes each batch re-uses the dof arrays
                # - Check if we need this ... we need this

                # Just calculate the amount of memory used at kernel generation time
                # and refuse to run the test if it uses more than the available amount
                for kio in k_inner_outer_options(n_in, kii, local_mem_size,# // n_dof_arrays,
                            fp_bytes=fp_bytes,start_val=kio_s,nelem=n_elem):
                    kio_s = None # Set to None so will form the full set the next time around
                    for iii in i_inner_inner_options(n_out, kii,
                            max_work_group_size=max_work_group_size, start_val=iii_s):
                        iii_s = None
                        for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                            iio_s = None
                            for ji in j_inner_options(n_in, start_val=ji_s):
                                ji_s = None
                                choices = (batch_size, kio, kii, iio, iii, ji)
                                parameter_list.append(choices)

    # Hacky, is there a better way to get this?
    within_inames, r_inames = list(get_einsum_types(knl))[0]
    for iname in r_inames:
        if "idof" in iname:
            j = iname
        elif "idim" in iname:
            r = iname
        elif "iface" in iname:
            f = iname
    for iname in within_inames:
        if "iel" in iname:
            e = iname
        elif "idof" in iname:
            i = iname
        elif "idim" in iname:
            x = iname


    # Now create the list of transformations instructions with those parameters

    #print(parameter_list)

    trans_list_list = []
    for params in parameter_list:

        trans_list = []
        batch_size, kio, kii, iio, iii, ji = params

        # TODO: Change this to a frozendict for easier legibility
        if not kio == 0 or iio == 0 or ji == 0 or iii == 0 or kii == 0: # If there is a zero length dimension then don't transform
            if kio != kii:
                trans_list.append(("split_iname", (f"{e}", kio,),
                    (("outer_tag", "g.0",), ("slabs",(0,1,),),),))
                trans_list.append(("split_iname", (f"{e}_inner", kii,), 
                    (("outer_tag", "ilp",), ("inner_tag", "l.0",), ("slabs", (0,1,),),),))
                prefetch_str = f"{j},{e}_inner_outer,{e}_inner_inner"
            else:
                trans_list.append(("split_iname", (f"{e}", kio,),
                    (("outer_tag", "g.0",), ("inner_tag", "l.0",), ("slabs",(0,0,),),),))
                prefetch_str = f"{j},{e}_inner"    
            if iio != iii:
                trans_list.append(("split_iname", (f"{i}", iio,),
                    (("outer_tag", "g.1",), ("slabs",(0,0,),),),))
                # In theory this should be (0,0)
                trans_list.append(("split_iname", (f"{i}_inner", iii,), 
                    (("outer_tag", "ilp",), ("inner_tag","l.1",), ("slabs",(0,0,),),),))
            else:
                trans_list.append(("split_iname", (f"{i}", iio,), 
                    (("outer_tag", "g.1",), ("inner_tag", "l.1",), ("slabs",(0,0,),),),))
            # Should the i loop have (0,1) slabs for both?

            if nr is not None:
                trans_list.append(("tag_inames", (((f"{r}", "unr",),),),))
            if nx is not None:
                if batch_size == 0:
                    # Breaks with einsum batching
                    trans_list.append(("tag_inames", (((f"{x}", "ilp",),),),))
                else:
                    trans_list.append(("tag_inames", (((f"{x}", "unr",),),),))
            if nf is not None:
                trans_list.append(("tag_inames", (((f"{f}", "unr",),),),))

            # The more einsums the slower the prefetching becomes
            for arg in read_dof_arrays:
                # Should only prefetch if there are no indirection arrays
                strides = [dim_tag.stride for dim_tag in arg_dict[arg].dim_tags if isinstance(dim_tag, lp.kernel.array.FixedStrideArrayDimTag)]
                order_str = "f,f" if strides[0] < strides[1] else "c,c"
                trans_list.append(("add_prefetch", (f"{arg}", prefetch_str,),
                    (("temporary_name", f"{arg}f",), ("default_tag","l.auto",),),))
                trans_list.append(("tag_array_axes", (f"{arg}f", order_str,),))

            for arg in face_dof_arrays:
                # Stick with the default ordering for now. For fortran ordering
                # slap an order tag on it.
                if kio != kii:
                    prefetch_str = f"{f},{j},{e}_inner_outer,{e}_inner_inner"
                else:
                    prefetch_str = f"{f},{j},{e}_inner"    

                trans_list.append(("add_prefetch", (f"{arg}", prefetch_str,),
                    (("temporary_name", f"{arg}f",), ("default_tag","l.auto",),),))

            trans_list.append(("split_iname", (f"{j}", ji,), (("outer_tag","for",), ("inner_tag","for",),),))

        trans_list.append(("add_inames_for_unused_hw_axes",))
        trans_list.append(("batch_einsums", (batch_size,),))
        trans_list_list.append(tuple(trans_list))

    print("Num trans to try: ", len(trans_list_list))

    return trans_list_list


# Is there any real reason to separate this from the tspace kernel. Why not
# just call this from within that function?
# Because the pspace is created from a subkernel, but the transformed
# kernel is the cumulative kernel.
def einsum3to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size

    # Find read dof arrays. These are the ones that will be prefetched
    read_deps = frozenset()
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            read_deps |= instr.dependency_names()

    dof_arrays = []
    from meshmode.transform_metadata import FirstAxisIsElementsTag
    for arg in knl.default_entrypoint.args + list(knl.default_entrypoint.temporary_variables.values()):
        if FirstAxisIsElementsTag() in arg.tags:
            dof_arrays.append(arg.name)
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif len(arg.shape) == 2: # The argument is (probably) an operator array
            n_out, n_in = arg.shape

    read_dof_arrays = read_deps & frozenset(dof_arrays)

    n_dof_arrays = len(read_dof_arrays)

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []

    if n_elem*n_out <= 1024:
        choices = (n_elem, n_elem, n_out, n_out, n_in)
        parameter_list.append(choices)
    else:
        for kii in k_inner_inner_options(start_val=kii_s):
            for kio in k_inner_outer_options(n_in, kii, local_mem_size // n_dof_arrays,
                        fp_bytes=fp_bytes,start_val=kio_s,nelem=n_elem):
                kio_s = None # Set to None so will form the full set the next time around
                for iii in i_inner_inner_options(n_out, kii,
                        max_work_group_size=max_work_group_size, start_val=iii_s):
                    iii_s = None
                    for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                        iio_s = None
                        for ji in j_inner_options(n_in, start_val=ji_s):
                            ji_s = None
                            choices = (kio, kii, iio, iii, ji)
                            parameter_list.append(choices)

    return parameter_list


def einsum2to2_kernel_tlist_generator_v2(queue, knl, start_param=None):

    # This type of einsum has no data reuse so could probably just
    # return a single set of transformations

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    # Find read dof arrays. These are the ones that will be prefetched
    read_deps = frozenset()
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            read_deps |= instr.read_dependency_names()

    dof_arrays = []
    arg_dict = dict([(arg.name, arg) for arg in knl.default_entrypoint.args])
    arg_dict.update(knl.default_entrypoint.temporary_variables)
    print(knl)
    for arg in list(arg_dict.values()):
        if len(arg.shape) == 2 and arg.name in read_deps:
            n_elem, n_in = arg.shape
            n_out = n_in
            fp_bytes = arg.dtype.dtype.itemsize

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s = (None, None, None, None)

    # Iterate over search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        for kio in k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    #for ji in j_inner_options(n_in, start_val=ji_s):
                    #    ji_s = None
                    choices = (kio, kii, iio, iii)
                    parameter_list.append(choices)
    
    # Get the element and dof inames
    for iname in knl.default_entrypoint.inames.keys():
        # Hacky, is there a better way to get this?
        if "iel" in iname:
            e = iname
        else:
            i = iname

    tlist_list = []
    for params in parameter_list:

        trans_list = []
        kio, kii, iio, iii = params
        #knl = kwargs["knl"]

        if 0 not in params: # If there is a zero length dimension then don't transform

            if kio != kii:
                trans_list.append(("split_iname", (f"{e}", kio,),
                    (("outer_tag", "g.0",), ("slabs",(0,1,),),),))
                trans_list.append(("split_iname", (f"{e}_inner", kii,),
                    (("outer_tag", "ilp",), ("inner_tag", "l.0",), ("slabs", (0,1,),),),))
            else:
                trans_list.append(("split_iname", (f"{e}", kio,),
                    (("outer_tag", "g.0",), ("inner_tag", "l.0",), ("slabs",(0,0,),),),))
            if iio != iii:
                trans_list.append(("split_iname", (f"{i}", iio,),
                    (("outer_tag", "g.1",), ("slabs",(0,0,),),),))
                trans_list.append(("split_iname", (f"{i}_inner", iii,),
                    (("outer_tag", "ilp",), ("inner_tag","l.1",), ("slabs",(0,1,),),),))
            else:
                trans_list.append(("split_iname", (f"{i}", iio,),
                    (("outer_tag", "g.1",), ("inner_tag", "l.1",), ("slabs",(0,0,),),),))

        """
        if knl.default_entrypoint.name == "resample_by_picking_group":
            trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["iel_inner", kii], 
                {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
            trans_list.append(["split_iname", ["idof_inner", iii], 
                {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
        else:
            trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["e_inner", kii], 
                {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
            trans_list.append(["split_iname", ["i_inner", iii], 
                {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
        """
        # Should the i loop have (0,1) slabs for both?

        # Prefetching probably matters not for this kernel
        #trans_list.append(["add_prefetch", ["arg1", "e_inner_outer,e_inner_inner,i_inner_outer,i_inner_inner"],
        #    {"temporary_name":"arg1f", "default_tag":"l.auto"}])
        #trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])

        trans_list.append(["add_inames_for_unused_hw_axes"])
        tlist_list.append(tuple(trans_list))

    return tlist_list



def einsum2to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii = params
    knl = kwargs["knl"]

    if knl.default_entrypoint.name == "resample_by_picking_group":
        trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["iel_inner", kii], 
            {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
        trans_list.append(["split_iname", ["idof_inner", iii], 
            {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    else:
        trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["e_inner", kii], 
            {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
        trans_list.append(["split_iname", ["i_inner", iii], 
            {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
        # Should the i loop have (0,1) slabs for both?

    # Prefetching probably matters not for this kernel
    #trans_list.append(["add_prefetch", ["arg1", "e_inner_outer,e_inner_inner,i_inner_outer,i_inner_inner"],
    #    {"temporary_name":"arg1f", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])

    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 


def einsum2to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    n_elem = None
    n_out = None
    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            if n_elem is None:
                n_elem, n_out = arg.shape
            else: # Needed to handle resample_by_picking_group
                n_elem = min(arg.shape[0], n_elem)
                n_out = min(arg.shape[1], n_out)
            n_in = n_out
            fp_bytes = arg.dtype.dtype.itemsize

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s = (None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        for kio in k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    #for ji in j_inner_options(n_in, start_val=ji_s):
                    #    ji_s = None
                    choices = (kio, kii, iio, iii)
                    parameter_list.append(choices)

    return parameter_list


def einsum4to2_face_mass_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params

    trans_list.append(["tag_inames", ["f: unr"]])
    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    trans_list.append(["add_prefetch", ["vec", "f,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"vecf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["vecf", "N2,N0,N1"]])

    trans_list.append(["add_prefetch", ["jac_surf", "f,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"jac_surff", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["jac_surff", "N2,N0,N1"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])

    trans_list.append(["add_inames_for_unused_hw_axes"]) 

    return trans_list 

"""
def einsum4to2_face_mass_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecDOFArray() in arg.tags:
            n_r, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape
        elif IsFaceMassOpArray() in arg.tags:
            n_out, n_r, n_in = arg.shape

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // (n_r + 1), fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list
"""


def einsum4to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]
    arg_names = {arg.name for arg in knl.default_entrypoint.args}
    inames = knl.default_entrypoint.inames.keys()
    
    if "r" in inames:
        trans_list.append(["tag_inames", ["r: unr"]])
    if "f" in inames:
        trans_list.append(["tag_inames", ["f: unr"]])
    

    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    #trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
    #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
    if "inv_jac_t" in arg_names:
        trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "N0,N1"]])
 
        trans_list.append(["add_prefetch", ["inv_jac_t", "r,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["inv_jac_tf", "N2,N0,N1"]])
    elif "jac_surf" in arg_names:
        trans_list.append(["add_prefetch", ["vec", "f,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
 
        trans_list.append(["add_prefetch", ["jac_surf", "f,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["inv_jac_tf", "N1,N0,N2"]])
 
    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum4to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    
    lmem_divisor = 0

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecDOFArray() in arg.tags:
            n_r, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsFaceDOFArray() in arg.tags:
            n_r, n_elem, n_in = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape
            lmem_divisor = n_r + 1
        elif IsFaceMassOpArray() in arg.tags:
            n_out, n_r, n_in = arg.shape
            lmem_divisor = 2*n_r

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // lmem_divisor, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


def einsum5to3_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    trans_list.append(["tag_inames", ["r: unr"]])
    trans_list.append(["tag_inames", ["x: ilp"]])
    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"vecf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
    trans_list.append(["add_prefetch", ["inv_jac_t", "x,r,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["inv_jac_tf", "N3,N2,N0,N1"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum5to3_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsFourAxisDOFArray() in arg.tags:
            n_r, n_x, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // (n_r*n_x + 1), fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list

