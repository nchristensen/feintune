import numpy as np
from pytools import memoize_in
from meshmode.array_context import EinsumTag

#import pyopencl as cl
#import pyopencl.array
#import pyopencl.clrandom

import loopy as lp
from grudge_tags import IsDOFArray, ParameterValue
#from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
#from loopy.kernel.data import AddressSpace

#import pycuda.gpuarray as cuarray
#import pycuda.driver as drv
#import pycuda.tools
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
#from pycuda.curandom import rand as curand

#from modepy import equidistant_nodes

#from bs4 import UnicodeDammit
import hjson
#import time
#from math import ceil
#import sys

# setup
# -----
lp.set_caching_enabled(False)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

# A lot of this could probably be deleted

def gen_face_mass_knl_merged(nelements, nfaces, nvol_nodes, nface_nodes, fp_format):
    knl =  lp.make_kernel(
         """{[iel,idof,fj]:
             0<=iel<nelements and
             0<=idof<nvol_nodes and
             0<=fj<nf_times_j}""",
         """
         result[iel,idof] = sum(fj, mat[idof, fj] * vec[iel, fj])
         """,
         kernel_data=[
             lp.GlobalArg("result", fp_format, shape=lp.auto, order="F"),
             lp.GlobalArg("vec", fp_format, shape=lp.auto, order="F"),
             lp.GlobalArg("mat", fp_format, shape=lp.auto, order="C"),
             "..."
         ],
         name="face_mass")

    # Gets around 470 GB/s
    knl = lp.fix_parameters(knl, nelements=nelements, nf_times_j=nfaces*nface_nodes, nvol_nodes=nvol_nodes)
    #knl = lp.tag_array_axes(knl, "result", "f,f")
    #knl = lp.tag_array_axes(knl, "vec", "f,f")

    knl = lp.split_iname(knl, "iel", 96, outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "iel_inner", 32, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
    knl = lp.add_prefetch(knl, "vec", "iel_inner_outer,iel_inner_inner,fj",
                            temporary_name="vecf", default_tag="l.auto")

    knl = lp.tag_array_axes(knl, "vecf", "f,f")
    knl = lp.split_iname(knl, "idof", 20, outer_tag="g.1", slabs=(0,0))
    knl = lp.split_iname(knl, "idof_inner", 2, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))
    knl = lp.split_iname(knl, "fj", 10, slabs=(0,0), inner_tag="unr")

    return knl


def gen_face_mass_knl(nelements, nfaces, nvol_nodes, nface_nodes, fp_format):
    knl =  lp.make_kernel(
         """{[iel,idof,f,j]:
             0<=iel<nelements and
             0<=f<nfaces and
             0<=idof<nvol_nodes and
             0<=j<nface_nodes}""",
         """
         #result[iel,idof] = sum(fj, mat[idof, fj] * vec[iel, fj])
         result[iel,idof] = sum(f, sum(j, mat[idof, f, j] * vec[f, iel, j]))
         """,
         kernel_data=[
             lp.GlobalArg("result", fp_format, shape=lp.auto),
             lp.GlobalArg("vec", fp_format, shape=lp.auto),
             lp.GlobalArg("mat", fp_format, shape=lp.auto),
             "..."
         ],
         name="face_mass")

    knl = lp.fix_parameters(knl, nelements=nelements, nfaces=nfaces, nvol_nodes=nvol_nodes, nface_nodes=nface_nodes)
    knl = lp.tag_array_axes(knl, "result", "f,f")
    knl = lp.tag_array_axes(knl, "vec", "N1,N0,N2")

    # Gets around 450 GB/s

    knl = lp.split_iname(knl, "iel", 96, outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "iel_inner", 32, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
    knl = lp.add_prefetch(knl, "vec", "j,iel_inner_outer,iel_inner_inner,f",
                            temporary_name="vecf", default_tag="l.auto")

    knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2")
    knl = lp.split_iname(knl, "idof", 20, outer_tag="g.1", slabs=(0,0))
    knl = lp.split_iname(knl, "idof_inner", 4, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))
    knl = lp.split_iname(knl, "j", 10, slabs=(0,0))

    return knl


def gen_elwise_linear_knl(n_elem, n_in, n_out, fp_format):

    knl = lp.make_kernel(
        """{[iel, idof, j]:
            0<=iel<nelements and
            0<=idof<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        "result[iel, idof] = sum(j, mat[idof, j] * vec[iel, j])",
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_elem, n_out), order="F"),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F"),
            lp.GlobalArg("mat", fp_format, shape=(n_out, n_in), order="C")    
        ],
        name="elwise_linear")
    knl = lp.fix_parameters(knl, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)


    #result = lp.tag_array_axes(result, "mat", "stride:auto,stride:auto")
    return knl

# Se podría usar el de Grudge.
#@memoize_method
def gen_diff_knl_fortran2(n_mat, n_elem, n_in, n_out, fp_format=np.float32,
        options=None):
    
    @memoize_in(gen_diff_knl_fortran2, "_gen_diff_knl")
    def _gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format):
        knl = lp.make_kernel(
        """{[imatrix,iel,idof,j]:
            0<=imatrix<nmatrices and
            0<=iel<nelements and
            0<=idof<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result[imatrix,iel,idof] = simul_reduce(sum, j, diff_mat[imatrix, idof, j] * vec[iel, j])
        """,
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_mat, n_elem, n_out),
                offset=lp.auto),
            lp.GlobalArg("diff_mat", fp_format, shape=(n_mat, n_out, n_in),
                order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F",
                offset=lp.auto),
            lp.ValueArg("nelements", tags=ParameterValue(n_elem)),
            lp.ValueArg("nmatrices", tags=ParameterValue(n_mat)),
            lp.ValueArg("ndiscr_nodes_out", tags=ParameterValue(n_out)),
            lp.ValueArg("ndiscr_nodes_in", tags=ParameterValue(n_in))
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0 and nmatrices > 0",
        options=options,
        name="diff_{}_axis".format(n_mat)
        )
        return knl

    knl = _gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format)

    # This should be in array context probably but need to avoid circular dependency
    # Probably should split kernels out of grudge_array_context
    knl = lp.tag_inames(knl, "imatrix: ilp")
    knl = lp.tag_array_axes(knl, "diff_mat", "sep,c,c")
    knl = lp.tag_array_axes(knl, "result", "sep,f,f")
    knl = lp.tag_array_axes(knl, "vec", "f,f")
    knl = lp.fix_parameters(knl, nmatrices=n_mat, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)
    return knl


# Is k x i in F layout equivalent to i x k in C layout?
# If so, can we just call the gen_diff_knl?
# Pretty sure it is...
def gen_diff_knl_fortran(n_elem, n_in, n_out, fp_format=np.float32, options=None):
    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result1[k,i] = simul_reduce(sum, j, mat1[i, j] * vec[k, j])
        result2[k,i] = simul_reduce(sum, j, mat2[i, j] * vec[k, j])
        result3[k,i] = simul_reduce(sum, j, mat3[i, j] * vec[k, j])
        """,
        kernel_data=[
            lp.GlobalArg("result1", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("result2", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("result3", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("mat1", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("mat2", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("mat3", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F",
                offset=lp.auto)
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        options=options,
        name="diff"

    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in,
        ndiscr_nodes_out=n_out)

    return knl

#@memoize_method
def gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format=np.float32, options=None):
    print(fp_format)
    knl = lp.make_kernel(
        """{[m,k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in and
            0<=m<nmatrices}""",
        """
        result[m, i ,k] = simul_reduce(sum, j, diff_mat[m, i, j] * vec[j, k])
        """,
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_mat, n_out, n_elem),
                offset=lp.auto),
            lp.GlobalArg("diff_mat", fp_format, shape=(n_mat, n_out, n_in),
                order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_in, n_elem), order="C",
                offset=lp.auto)
        ],
        #kernel_data = [
        #    lp.GlobalArg("result1", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("result2", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("result3", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("mat1", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("mat2", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("mat3", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("vec", fp_format, shape=None, strides=(1, n_elem),
        #       offset=lp.auto, order="C")
        #],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0 \
                     and nmatrices > 0",
        options=options,
        name="diff"
    )
    knl = lp.tag_array_axes(knl, "diff_mat", "sep,c,c")
    knl = lp.tag_array_axes(knl, "result", "sep,c,c")
    knl = lp.tag_array_axes(knl, "vec", "c,c")

    knl = lp.fix_parameters(knl, nmatrices=n_mat, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)

    #mat_string = ["result1", "result2", "result3", "vec"]
    #for i in range(len(mat_string)):
    #   knl = lp.tag_array_axes(knl, mat_string, "stride:auto,stride:auto")
    #   knl = lp.tag_array_axes(knl, mat_string, "N1,N0")

    return knl


# This is redundant with the above but is more clear than the above
# so to keep it around may be worthwhile.
'''
def gen_diff_knl(n_elem, n_in, n_out, k_inner_outer,k_inner_inner,i_inner_outer,
                    i_inner_inner,j_inner, fp_format=np.float32):
    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result1[i,k] = simul_reduce(sum, j, mat1[i, j] * vec[j, k])
        result2[i,k] = simul_reduce(sum, j, mat2[i, j] * vec[j, k])
        result3[i,k] = simul_reduce(sum, j, mat3[i, j] * vec[j, k])
        """,
        kernel_data = [
            lp.GlobalArg("result1", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("result2", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("result3", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("mat1", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("mat2", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("mat3", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("vec", fp_format, shape=(n_in, n_elem), order="C")
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        default_offset=None,
        name="diff"
    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in,
                                ndiscr_nodes_out=n_out)

    slabs0 = (0,0) if n_elem % k_inner_outer == 0 else (0,1)
    knl = lp.split_iname(knl, "k", k_inner_outer, outer_tag="g.0", slabs=slabs0)
    knl = lp.split_iname(knl, "k_inner", k_inner_inner, outer_tag="ilp",
                            inner_tag="l.0")
    knl = lp.split_iname(knl, "j", j_inner)
    knl = lp.split_iname(knl, "i", i_inner_outer, outer_tag="g.1")#slabs=(0,1))
    knl = lp.split_iname(knl, "i_inner", i_inner_inner, outer_tag="ilp",
                            inner_tag="l.1")

    #knl = lp.prioritize_loops(knl, "j_outer,j_inner,k_inner_outer")

    knl = lp.add_prefetch(knl, "vec", "j_outer,j_inner,k_inner_outer,k_inner_inner",
                            temporary_name="vecf", default_tag="l.auto")
    knl = lp.add_prefetch(knl, "mat1", "j_inner", temporary_name="mat1fp",
                            default_tag="unr")
    knl = lp.add_prefetch(knl, "mat2", "j_inner", temporary_name="mat2fp",
                            default_tag="unr")
    knl = lp.add_prefetch(knl, "mat3", "j_inner", temporary_name="mat3fp",
                            default_tag="unr")

    return knl
'''


def load_transformations_from_file(hjson_file, indices): 
    od = hjson.loads(hjson_file.read())
    for index in indices:
        od = od[index]
    return od

def generate_transformation_list_old(k_inner_outer, k_inner_inner, i_inner_outer,
                                    i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args
    transformations.append(("split_iname", ["k", k_inner_outer], {"outer_tag": "g.0",
                                "slabs": (0, 1)}))
    transformations.append(("split_iname", ["k_inner", k_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["j", j_inner]))
    transformations.append(("split_iname", ["i", i_inner_outer],
                            {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["i_inner", i_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("add_prefetch", ["vec",
                            "j_outer,j_inner,k_inner_outer,k_inner_inner"],
                            {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("add_prefetch", ["mat1", "j_inner"],
                            {"temporary_name": "mat1fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat2", "j_inner"],
                            {"temporary_name": "mat2fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat3", "j_inner"],
                            {"temporary_name": "mat3fp", "default_tag": "unr"}))
    return tuple(transformations)

# This is rather nvidia specific at present
# And also specific to the diff kernel
# May need different ones of these for different kernels
def generate_transformation_list(k_inner_outer, k_inner_inner, i_inner_outer,
                                i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args

    # Set data layouts
    # This should be handled by the array context?
    #transformations.append(("tag_array_axes", ["diff_mat", "sep,c,c"]))
    #transformations.append(("tag_array_axes", ["result", "sep,f,f"]))

    # Split and tag inames
    #transformations.append(("tag_inames", [[("imatrix", "ilp")]]))
    transformations.append(("split_iname", ["iel", k_inner_outer], {"outer_tag": "g.0",
                            "slabs": (0, 1)}))
    transformations.append(("split_iname", ["iel_inner", k_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["idof", i_inner_outer],
                            {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["idof_inner", i_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("split_iname", ["j", j_inner]))

    # Prefetching
    transformations.append(("add_prefetch", ["vec",
                            "j_outer,j_inner,iel_inner_outer,iel_inner_inner"],
                            {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("tag_array_axes", ["vecf", "f,f"]))
    transformations.append(["add_inames_for_unused_hw_axes"])
    return tuple(transformations)

def get_einsums(knl):
    einsums = []
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            for tag in instr.tags:
                if isinstance(tag, EinsumTag):
                    if isinstance(instr.expression, lp.symbolic.Reduction):
                        einsums.append((instr.within_inames, instr.expression.inames,))
                    else:
                        einsums.append((instr.within_inames, (),))
                    
    #print(knl.default_entrypoint.name, einsums)
    return einsums

def get_einsum_counts(knl):
    from collections import Counter
    counter = Counter(get_einsums(knl))
    #print(counter)
    return counter

# Obtain non-reduction and reduction inames 
def get_einsum_types(knl):
    return frozenset(get_einsums(knl))



def add_batch_ids(tunit, batch_size):
    from meshmode.array_context import EinsumTag
    assert batch_size >= 0
    new_instructions = []
    batch_number = 0
    used_batches = 0
    num_in_cur_batch = 0
    batch_instructions_list = []
    # Could add some priority level based on the length of the chain of einsums so 
    # if there is a dependency chain es1 -> es2 -> es3 then es2 must be put in
    # a batch higher than that of es1 and ditto with es3
    batch_instructions = []
    for instr in tunit.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment) and any([isinstance(tag, EinsumTag) for tag in instr.tags]):
            new_instr = instr.copy(id=f"batch_{batch_number}_" + instr.id)
            new_instructions.append(new_instr)
            batch_instructions.append(new_instr)
            num_in_cur_batch += 1
            
            if num_in_cur_batch == batch_size:
                batch_number += 1
                num_in_cur_batch = 0
                batch_instructions_list.append(batch_instructions)
                batch_instructions = []
        else:
            new_instructions.append(instr)

    # Handle a non-full final batch
    if len(batch_instructions_list) > 0 and len(batch_instructions_list) < batch_size:
        batch_instructions_list.append(batch_instructions)

    # Need to add batch ids to the prefetch instructions
    # and order the batches. The prefetching of the next batch can't begin until the current batch finishes
    # Can us the group numbers to order tmpgrp__actx_in_1_0_momentum_1_0f

    return tunit.with_kernel(tunit.default_entrypoint.copy(instructions=new_instructions)), batch_instructions_list


# Will the temporaries group automatically handle the einsum chunking problem?
#def alias_temporaries_among_batches(tunit, nbatches):
#    batch_instructions = {}
#    for instr in tunit.default_entrypoint.instructions:
        
    # Just get the temporaries of each batch and alias those of the same size

"""
def get_batch_temporaries_by_size(tunit, batches):

    # Assumes all of the temporaries are in local or private memory
    temp_dict = tunit.default_entrypoint.temporary_variables
    batch_dict_list = [] # A list of dictionaries of sets

    for batch in batches:
        batch_dict = {}
        for einsum in batch:
            for dep in einsum.dependency_names():
                if dep in temp_dict:
                    shape = temp_dict[dep].shape
                    if shape not in batch_dict:
                        batch_dict[shape] = set([dep])
                    else:
                        batch_dict[shape] |= set([dep])
        batch_dict_list.append(batch_dict)

    return batch_dict_list
"""

def get_batch_temporaries_by_size(tunit, nbatches, address_space=None):

    # Assumes all of the temporaries are in local or private memory
    #for key, val in tunit.default_entrypoint.temporary_variables.items():
    #    print(key, val.address_space)
    #exit()
    if address_space is None:
        temp_dict = tunit.default_entrypoint.temporary_variables
    else:
        temp_dict = {key: val for key, val in tunit.default_entrypoint.temporary_variables.items() if val.address_space == address_space}
    #print(temp_dict)
    #exit()
    batch_dict_list = [] # A list of dictionaries (keyed by size, one for each batch) of sets of temporary ids

    # Inefficient
    for batch_num in range(nbatches):
        batch_dict = {}
        for instr in tunit.default_entrypoint.instructions:
            if f"batch_{batch_num}" in instr.id:
                for dep in instr.dependency_names():
                    if dep in temp_dict:
                        shape = temp_dict[dep].shape
                        if shape not in batch_dict:
                            batch_dict[shape] = set([dep])
                        else:
                            batch_dict[shape] |= set([dep])
                           
        batch_dict_list.append(batch_dict)

    return batch_dict_list


def get_alias_sets(batch_dict_list):
    print(batch_dict_list)
    sizes = set()
    for batch_dict in batch_dict_list:
        sizes |= set(batch_dict.keys())
    
    alias_sets = []
    for size in sizes:
        arg_lists = []
        for batch_dict in batch_dict_list:
            arg_lists.append(sorted(batch_dict[size]))

        max_len = 0
        for l in arg_lists:
            max_len = max(len(l), max_len)
        for l in arg_lists:
            while len(l) < max_len:
                l.append(None) # Pad with None so can slice columns

        # This needs to be a bit more robust, we want a variable to alias with itself.
        # We also can't allow two tempories to alias if they occur in the same block (row)

        # Permute so if a value is found in the current row it isn't found in the current column
        # except if the value is self
        # Permute so self is in current column as much as possible
        # For now just assert to verify this is not the case, don't attempt to fix
        arg_array = np.array(arg_lists)

        flat_arg_array = arg_array.flatten()
        nonzero_entries = flat_arg_array[np.flatnonzero(flat_arg_array)]
        unique_entries = np.unique(nonzero_entries)
        assert len(unique_entries) == len(nonzero_entries)

        for col in range(arg_array.shape[1]):
            col_set = set(arg_array[:,col].flatten())
            for row in range(arg_array.shape[0]):
                row_set = set(arg_array[row,:].flatten())
                assert col_set & row_set == set([arg_array[row,col]])

        for col in range(arg_array.shape[1]):
            alias_sets.append(set(arg_array[:,col]) - set([None]))
    
    return alias_sets


def batch_einsums(tunit, batch_size, **kwargs):
    if batch_size == 0:
        return tunit

    # Need to get the existing tags and apply them to the new loops
    #orig_inames = tunit.default_entrypoint.inames.keys()
    orig_nonglobal_inames = []
    # Will need to think about the names of prefetching arrays
    for name, iname in tunit.default_entrypoint.inames.items():
        if not any([isinstance(tag, lp.kernel.data.GroupInameTag) for tag in iname.tags]):
            orig_nonglobal_inames.append(name)
    #print(orig_nonglobal_inames)

    #orig_iname_dict = tunit.default_entrypoint.inames.items()
    b_tunit, batches = add_batch_ids(tunit, batch_size)
    nbatches = len(batches)
    knl = b_tunit.default_entrypoint

    #print(knl)
    #print("AFTER PREPROCESS")
    #print(lp.preprocess_kernel(tunit))
    #exit()

    def linearize_batches(tunit, batches):

        knl = tunit.default_entrypoint
        nbatches = len(batches)

        # Alias local memory temporaries
        #batch_temps_by_size = get_batch_temporaries_by_size(tunit, nbatches) 
        #alias_sets = get_alias_sets(batch_temps_by_size)
        #for s in alias_sets:
        #    knl = lp.alias_temporaries(knl, list(s))

        # Map instruction ids to fetch rules
        fetch_rules = set([instr.id for instr in knl.instructions if "fetch_rule" in instr.id])
        for i, batch in enumerate(batches):
            for einsum in batch:
                fetch_rule_deps = einsum.depends_on & fetch_rules
                if len(fetch_rule_deps) > 0 and i > 0:
                    for fetch_rule in fetch_rule_deps:
                        # Make the fetch rule depend on the immediately prior batch if no prior batch already depends on it
                        add_dep = True
                        for j in range(i-1, -1, -1):
                            if any([fetch_rule in prior_batch_einsum.depends_on for prior_batch_einsum in batches[j]]):
                                add_dep = False
                                break

                        if add_dep:
                            j = i - 1
                            knl = lp.add_dependency(knl, f"id:{fetch_rule}", f"id:batch_{j}*")
                        

        # Create independent loops for each batch                
        for i in range(0, nbatches): # Should we keep the first batch in the original set of loops
            # Can the we not duplicate the inames with global tags?
            knl = lp.duplicate_inames(knl, orig_nonglobal_inames, f"id:batch_{i}_*")
        # Transfer the tags to the new kernel
        iname_dict = knl.inames
        new_iname_dict = knl.inames.copy()

        # O(n²) complexity. This could be more efficient
        for name, iname in knl.inames.items():
            for old_name, old_iname in tunit.default_entrypoint.inames.items():
                # If the iname prefix is the same, then apply the old tags
                if old_name in name and len(iname.tags) == 0:
                    new_iname_dict[name] = iname.tagged(old_iname.tags)

        knl = knl.copy(inames=new_iname_dict) 

        # Alias private memory temporaries (for some reason nothing is known about the local memory
        # temporaries if we attempt to alias them here too)
        #b_tunit = lp.preprocess_program(tunit.with_kernel(knl)) # Realizes the reductions so we can access the accumulators
        from loopy.transform.realize_reduction import realize_reduction
        b_tunit = realize_reduction(tunit.with_kernel(knl), unknown_types_ok=True)
        batch_temps_by_size = get_batch_temporaries_by_size(b_tunit, nbatches) 
        alias_sets = get_alias_sets(batch_temps_by_size)
        #print(alias_sets)
        #exit()
        knl = b_tunit.default_entrypoint
        for s in alias_sets:
            knl = lp.alias_temporaries(knl, list(s))

        

        """
        print("Old kernel")
        for name, iname in tunit.default_entrypoint.inames.items():
            print(name, iname.tags)
        print("New Kernel")
        for name, iname in knl.inames.items():
            print(name, iname.tags)
        """     

        #print(knl)
        #exit()

        return b_tunit.with_kernel(knl)

    b_tunit = linearize_batches(b_tunit, batches)

    return b_tunit

    #return tunit.with_kernel(knl)


def apply_transformation_list(knl, transformations):
    # Could just construct a string for the function handle and retrieve the function from that
    function_mapping = {"split_iname": lp.split_iname,
                        "add_prefetch": lp.add_prefetch,
                        "prioritize_loops": lp.prioritize_loops,
                        "rename_iname": lp.rename_iname,
                        "tag_array_axes": lp.tag_array_axes,
                        "tag_inames": lp.tag_inames,
                        "batch_einsums": batch_einsums,
                        "add_inames_for_unused_hw_axes": lp.add_inames_for_unused_hw_axes}

    # Maybe add some logic to add slabs=(0,0) if n_elem % k_inner_outer == 0
    # Maybe can do this based on tranformation name, loop variable, and loop variable
    # bounds
    print(knl.default_entrypoint)
    print(transformations)
    #exit()
    for t in list(transformations):
        print(t)
        func = function_mapping[t[0]]
        args = [knl]
        if len(t) > 1:
            args = args + list(t[1])
        kwargs = dict(t[2]) if len(t) > 2 else {}
        knl = func(*args, **kwargs)
    return knl
