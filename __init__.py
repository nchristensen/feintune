import numpy as np
from pytools import memoize_in
from meshmode.array_context import EinsumTag
from decouple_domain import decouple_domain
from utils import get_domain_list
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
import time
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
    assert batch_size >= 1
    # Should a batch size of zero be equal to a single batch
    new_instructions = []
    batch_number = 0
    used_batches = 0
    num_in_cur_batch = 0
    batch_instructions_list = []
    # Could add some priority level based on the length of the chain of einsums so 
    # if there is a dependency chain es1 -> es2 -> es3 then es2 must be put in
    # a batch higher than that of es1 and ditto with es3
    batch_instructions = []
    insn_mappings = {}
    for instr in tunit.default_entrypoint.instructions:
        # If collect the einsums in a list then the batch number should be index // batch_size?
        # The below seems unneccessarily complicated.

        if isinstance(instr, lp.Assignment) and any([isinstance(tag, EinsumTag) for tag in instr.tags]):
            insn_mappings[instr.id] = [f"batch_{batch_number}_" + instr.id]

            # Also add batch prefix to any prefetch instructions for the einsum
            for dep_id in instr.depends_on:
                if "fetch_rule" in dep_id:
                    insn_mappings[dep_id] = [f"batch_{batch_number}_" + dep_id]

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
    if len(batch_instructions) > 0 and len(batch_instructions) < batch_size:
        batch_instructions_list.append(batch_instructions)

    # Need to add batch ids to the prefetch instructions
    # and order the batches. The prefetching of the next batch can't begin until the current batch finishes
    # Can us the group numbers to order tmpgrp__actx_in_1_0_momentum_1_0f

    """
    fetch_rules = set([instr.id for instr in tunit.default_entrypoint.instructions if "fetch_rule" in instr.id])

    for i, batch in enumerate(batch_instructions_list):
        for einsum in batch:
            fetch_rule_deps = einsum.depends_on & fetch_rules
            for fetch_rule in fetch_rule_deps:
                # Assume fetch_rules only fetch for a single einsum
                assert fetch_rule not in fetch_rule_mapping 
                fetch_rule_mapping[fetch_rule] = fetch_rule.copy(id="batch_{i}_" + instr.id)
    """

    #print(insn_mappings)
    #print(tunit.default_entrypoint)
    new_knl = lp.replace_instruction_ids(tunit.default_entrypoint, insn_mappings)
    #for instr in tunit.default_entrypoint.instructions:
    #    print(instr)
    #print()
    #for instr in new_knl.instructions:
    #    print(instr)
    #print(new_knl)
    #exit()

    return tunit.with_kernel(new_knl), batch_instructions_list # Maybe don't need the batch instructions list anymore?
                
    #for i, instr in enumerate(new_instructions):
    #    if instr in fetch_rule_mapping:
    #        print("HERE")
    #        new_instructions[i] = fetch_rule_mapping[instr]

    #exit()

    #return tunit.with_kernel(tunit.default_entrypoint.copy(instructions=new_instructions)), batch_instructions_list


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

# TODO: Make data type an argument and only alias for a single data type at a time.
# For now, assume all temporaries have the same data type.
def get_batch_temporaries_by_size(tunit, nbatches, address_space):

    # Assumes all of the temporaries are in local or private memory
    temp_dict = {key: val for key, val in tunit.default_entrypoint.temporary_variables.items() if val.address_space==address_space}
    #print("Temp dict:", temp_dict)
    #exit()
    batch_dict_list = [] # A list of dictionaries (keyed by size, one for each batch) of sets of temporary ids

    # Inefficient
    for batch_num in range(nbatches):
        batch_dict = {}
        for instr in tunit.default_entrypoint.instructions:
            if f"batch_{batch_num}" in instr.id:
                for dep in instr.dependency_names():
                    if dep in temp_dict:
                        size = np.product(temp_dict[dep].shape)
                        if size not in batch_dict:
                            batch_dict[size] = set([dep])
                        else:
                            batch_dict[size] |= set([dep])
                           
        batch_dict_list.append(batch_dict)

    return batch_dict_list


def get_alias_sets(batch_dict_list):
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
        print(unique_entries)
        print(nonzero_entries)
        assert len(unique_entries) == len(nonzero_entries)

        for col in range(arg_array.shape[1]):
            col_set = set(arg_array[:,col].flatten())
            for row in range(arg_array.shape[0]):
                row_set = set(arg_array[row,:].flatten())
                assert col_set & row_set == set([arg_array[row,col]])

        for col in range(arg_array.shape[1]):
            alias_sets.append(set(arg_array[:,col]) - set([None]))
    
    return alias_sets


from qprofile import qprofile, qinstrument

# Should probably be renamed batch_einsums_and_prefetch or similar
# these transformations seem to be linked
#@qprofile
#@qinstrument
def batch_einsums(tunit, batch_size, **kwargs):
    from pyinstrument import Profiler
    profiler=Profiler()
    profiler.start()
    
    print("BATCHING THE EINSUMS")
    #exit()

    # Or if the batch size is greater than the number of einsums?
    if batch_size <= 0:
        return tunit

    #print(tunit)
    #exit()

    # Need to get the existing tags and apply them to the new loops
    # Maybe have the option of batching by global inames or local inames.
    # Might be less cache pollution if the batching is done at the global level.
    orig_nonglobal_inames = []
    #orig_inames = tunit.default_entrypoint.inames.items()
    # Will need to think about the names of prefetching arrays
    for name, iname in tunit.default_entrypoint.inames.items():
        if not any([isinstance(tag, lp.kernel.data.GroupInameTag) for tag in iname.tags]): 
            orig_nonglobal_inames.append(name)
    #inames_to_duplicate = sorted(inames_to_duplicate)
    #inames_to_duplicate = sorted(tunit.default_entrypoint.inames.keys())
    #print(len(inames_to_duplicate))
    #inames_to_duplicate = sorted(inames_to_duplicate + [iname for iname in tunit.default_entrypoint.inames.keys() if "iel_" in iname])
    #for iname in inames_to_duplicate:
    #    print(iname)
    #print(tunit)
    #exit()

    non_fetch_rule_inames = set()
    for instr in tunit.default_entrypoint.instructions:
        if not "fetch_rule" in instr.id: # Could be a problem if someone overrides the inames at any point
            non_fetch_rule_inames |= set(instr.within_inames)

    within_inames = set()
    for instr in tunit.default_entrypoint.instructions:
            within_inames |= set(instr.within_inames)

    additional_inames_to_duplicate = set(tunit.default_entrypoint.inames.keys()) - within_inames

    # Need to rename this variable
    #print(non_fetch_rule_inames)
    #inames_to_duplicate = sorted([iname for iname in tunit.default_entrypoint.inames.keys() if not "actx_" in iname])
    #inames_to_duplicate = sorted(non_fetch_rule_inames) # Apparently this does not include idof_ensm1
    #print(inames_to_duplicate)

    #print(tunit.default_entrypoint.inames.keys())
    #print(non_fetch_rule_inames)
    #exit()
        
            
            

    #fetch_rules = [instr for instr in knl.instructions if "fetch_rule" in instr.id]
    #fetch_rule_union_inames = 
    #fetch_rule_intersection_inames

    # Do we really need to copy the actx_* inames? Can those go away?
    #print(inames_to_duplicate)
    #exit()

    #inames_to_duplicate = sorted(tunit.default_entrypoint.inames.keys()) # Override to global for now
    #print(inames_to_duplicate)

    #orig_iname_dict = tunit.default_entrypoint.inames.items()

    # Returning the batches is now unnecessary
    b_tunit, batches = add_batch_ids(tunit, batch_size)
    nbatches = len(batches)
    knl = b_tunit.default_entrypoint

    #print(knl)
    #exit()

    #print(knl)
    #print("AFTER PREPROCESS")
    #print(lp.preprocess_kernel(tunit))

    # Attempt to avoid the poor scaling of add_prefetch by applying the
    # prefetching to subkernels which are then recombined
    # prefetch_data is a list of tuples of (argname, prefetch_str)

    """
    def add_prefetches_by_batch(tunit, nbatches, prefetch_data, **kwargs):

        # Seems like there are two tasks, one is getting the batch/phase
        # instructions, which is particular to desired decomposition
        # then, creating the subkernels with those instructions, which
        # is rather generic and code can probably be re-used there.
        # Obtaining the instruction domains can be done in the generic part.
        # Maybe the latter part can go into Loopy.

        # Could the testing be done using a single batch instead of the full kernel?

        # Create the batch subkernels
        def create_batch_subkernels(tunit, nbatches):
            batch_instructions = [[]]*nbatches
            batch_domains = [[]]*nbatches
            for instr in tunit.default_entrypoint.instructions:
                if "batch_" in instr.id:
                    batch_num = instr.id.split("_")[1]
                    batch_instructions[batch_num].append(instr)
                    batch_domains[batch_num].append(instr.within_inames)
                
            # Replace inames with domain objects                     

            # Move this function out of init
            domain_list = get_domain_list(tunit)
            for batch_num in nbatches:
                domain_ids = batch_domains[batch_num]
                batch_domains[batch_num] = []
                
                for domain_names_set, domain in domain_list:
                    if domain_ids <= domain_names_set:
                        batch_domains[batch_num].append(domain)

            for batch_num in nbatches:
                domains = batch_domains[batch_num]
                instructions = batch_instructions[batch_num]
                 
           # Can re-use some code from generate_subkernels
           # In fact, phases are basically the same thing as batches,
           # though they are currently indexed by barrier rather than
           # a phase/batch number 
                

        # Figure out what prefetches apply to what batches
        # May as well tag the prefetch instructions with the batch numbers as well

        # Apply the prefetches in the batches

        # Combine the subkernels

        # Proceed to batching

    """


    # Maybe this needs to be a separate transformation
    def linearize_batches(tunit, batches):
        print("Linearizing batches")

        knl = tunit.default_entrypoint
        nbatches = len(batches)

        # See if stripping off the iname tags makes batching faster
        # -- It doesn't seem to help
        """
        iname_tags_dict = {}
        for iname_name, iname_obj in knl.inames.items():
            iname_tags_dict[iname_name] = iname_obj.tags

        tagless_inames = {}
        for iname_name, iname_obj in knl.inames.items():
            tagless_inames[iname_name] = iname_obj.copy(tags=frozenset())

        print(knl)
        knl = knl.copy(inames=tagless_inames)
        print(knl)
        exit()
        """

        # Alias local memory temporaries
        #batch_temps_by_size = get_batch_temporaries_by_size(tunit, nbatches) 
        #alias_sets = get_alias_sets(batch_temps_by_size)
        #for s in alias_sets:
        #    knl = lp.alias_temporaries(knl, list(s))

        # Map instruction ids to fetch rules, will probably need to add this part to prefetch_and_project too
        """
        fetch_rules = set([instr.id for instr in knl.instructions if "fetch_rule" in instr.id])

        #print(knl)
        #exit()
        #kern = knl.copy(target=lp.CTarget())
        #kern = b_tunit.copy(target=lp.OpenCLTarget())
        #code = lp.generate_code_v2(kern).device_code()
        #print(code)
        #exit()
        #print(type(batches))
        #exit()

        print("Adding dependencies")

        for i, batch in enumerate(batches[1:],start=1):

            '''
            for einsum in batch:
                j = i - 1
                knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")                

                # The following is needed if batching occurs after prefetching
                # Note that this will make the code non-generatable until the batch inames are duplicated
                fetch_rule_deps = einsum.depends_on & fetch_rules
                for fetch_rule in fetch_rule_deps:
                    # Make the fetch rule depend on the immediately prior batch if no prior batch already depends on it
                    add_dep = True
                    for k in range(i-1, -1, -1):
                        if any([fetch_rule in prior_batch_einsum.depends_on for prior_batch_einsum in batches[k]]):
                            add_dep = False
                            break

                    if add_dep:
                        knl = lp.add_dependency(knl, f"id:{fetch_rule}", f"id:batch_{j}_*")
            '''
        """
        # Enforcing an ordering may or may not reduce scheduling time
        # Actually, is needed for aliasing
        for i in range(1, nbatches):
            j = i - 1
            knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")

        #print(knl)
        #exit()
        #kern = knl.copy(target=lp.CTarget())
        #kern = b_tunit.copy(target=lp.OpenCLTarget())
        #code = lp.generate_code_v2(kern).device_code()
        #print(code)
        #exit()


        # Create independent loops for each batch                

        import time
        orig_inames = set(knl.inames.keys())

        #print(knl)

        # Decoupling with a frozenset puts each iname in its own set.
        #print("Decoupling all inames")
        #for iname in knl.inames.keys():
        #    print("Decoupling", iname)
        #    knl = decouple_domain(knl, iname, frozenset())
        print("DECOUPLING")
        # Decoupling the original inames seems to cause code generation problems,
        # but if this isn't done then decoupling takes forever
        #print(set(knl.inames.keys()) - set(inames_to_duplicate))
        #exit()
        
        #print(inames_to_duplicate)
        #exit()
        #knl = decouple_domain(knl, inames_to_duplicate[0:3], frozenset())

        #knl = decouple_domain(knl, knl.inames, frozenset())
        #knl = decouple_domain(knl, inames_to_duplicate, knl.inames.keys())
        #"""

        print("Duplicating inames")
        for i in range(0, nbatches): # Should we keep the first batch in the original set of loops?
            start = time.time()
            before_inames_dict = knl.inames.copy()

            batch_inames = set()
            for instr in knl.instructions:
                if f"batch_{i}_" in instr.id:
                    batch_inames |= instr.within_inames

            # For some reason idof_ensm2 does not appear in batch_inames. (Maybe it is a reduction iname?)
            # This is a hack to make sure it appears.
            batch_inames_to_duplicate = batch_inames | additional_inames_to_duplicate

            suffix = f"_b{i}"
            print("HERE")
            #knl = lp.duplicate_inames(knl, inames_to_duplicate, f"id:batch_{i}_*", suffix=suffix)
            knl = lp.duplicate_inames(knl, batch_inames_to_duplicate, f"id:batch_{i}_*", suffix=suffix)
            print("DONE HERE")

            after_inames_dict = knl.inames.copy()
            added_inames = set(after_inames_dict.keys()) - set(before_inames_dict.keys())


            # Orig nonglobal_inames may be too big a set
            # The problem is that prefetching adds a bunch of new nonglobal inames
            # Need to limit this to only the inames duplicated
            #parent_inames = set(inames_to_duplicate) | added_inames
            #parent_inames = set(batch_inames_to_duplicate) | added_inames

           
            #new_inames = sorted(added_inames)
           
            # Copy the tags
            # Make sure both iname lists are sorted here.

            #print(sorted(list(zip(inames_to_duplicate, new_inames))))
            #exit()
 
            if True:
                # Copy iname tags and slab increments, which are apparently dropped by default
                after_iname_slab_increments = dict(knl.iname_slab_increments)
                for old_iname in batch_inames_to_duplicate:#inames_to_duplicate:
                    new_iname = old_iname + suffix
                    if old_iname in after_iname_slab_increments:
                        after_iname_slab_increments[new_iname] = after_iname_slab_increments[old_iname]
                    after_inames_dict[new_iname] = after_inames_dict[new_iname].tagged(before_inames_dict[old_iname].tags)
                # Need to see of https://github.com/inducer/loopy/blob/32ce1373688383f69c2dee5d9e5c5f2bbe716867/loopy/codegen/loop.py#L317
                # can be removed.
                knl = knl.copy(inames=after_inames_dict, iname_slab_increments=after_iname_slab_increments) 


            end = time.time()
            print("Duplication time", i+1, end - start)


            start = time.time()

            print(added_inames)
            #exit()
            #inames_to_decouple = frozenset([iname for iname in added_inames if "iel_" not in iname])
            #print(inames_to_decouple)
            #exit()
            #knl = decouple_domain(knl, inames_to_decouple, frozenset())

            knl = decouple_domain(knl, added_inames, frozenset())
            before_removal = knl.inames.keys()
            knl = lp.remove_unused_inames(knl)
            after_removal = knl.inames.keys()
            print("Removed inames")
            print(set(before_removal) - set(after_removal))
            #exit()

            #knl = decouple_domain(knl, added_inames | orig_inames, frozenset())
            #knl = decouple_domain(knl, orig_inames, frozenset())
            #l_added_inames = list(added_inames)[:1]
            #for iname in l_added_inames:
            #   knl = decouple_domain(knl, [iname], frozenset())
            #knl = decouple_domain(knl, set(list(added_inames)[-1]), frozenset())
            # Cost grows quadratically
            #knl = decouple_domain(knl, knl.inames, frozenset())
            #for entry in added_inames:
            #    knl = decouple_domain(knl, [entry], set())   
            #knl = decouple_domain(knl, added_inames, set())
            #knl = decouple_domain(knl, orig_inames, set())
            end = time.time()
            print("Decoupling time", i+1, end-start)
            #exit()
                


            # Maybe we can just move the duplicated inames to a separate basic set.
            #if end - start > 20:
                #import pdb; pdb.set_trace()
                #return

            #knl = lp.duplicate_inames(knl, orig_inames, f"id:batch_{i}_*")

        #print(knl)
        #exit()
        #"""

        #"""
        # Transfer the tags to the new kernel. Is this redundant with the above tag copying?
        iname_dict = knl.inames
        new_iname_dict = knl.inames.copy()

        # O(n²) complexity. This could be more efficient
        new_inames = set(knl.inames.keys()) - orig_inames
        

        if False: # Domain decoupling screws with this... or maybe not. Might not be needed in any case.
            for name, iname in knl.inames.items():
                for old_name, old_iname in tunit.default_entrypoint.inames.items():
                    # If the iname prefix is the same, then apply the old tags
                    if old_name in name and len(iname.tags) == 0:
                        new_iname_dict[name] = iname.tagged(old_iname.tags)
            knl = knl.copy(inames=new_iname_dict) 


        """
        # Alias private memory temporaries (for some reason nothing is known about the local memory
        # temporaries if we attempt to alias them here too)
        # yes, because their memory spaces are lp.auto, need to preprocess to obtain them
        #"""

        if True:
            from loopy.transform.realize_reduction import realize_reduction

            pp_tunit = lp.preprocess_program(tunit.with_kernel(knl)) # Realizes the reductions so we can access the accumulators
            # realize_reduction doesn't fill in lp.auto memory spaces
            #pp_tunit = realize_reduction(tunit.with_kernel(knl), unknown_types_ok=True)
            #knl = b_tunit.default_entrypoint

            # This needs to be fixed. It prevents finding a successful schedule
            # Need to separate by address space. Or maybe just call twice?

            batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.LOCAL) 
            alias_sets = get_alias_sets(batch_temps_by_size)
            for s in alias_sets:
                knl = lp.alias_temporaries(knl, list(s))

            # Doesn't seem to do anything for OpenCL but for other targets it might do something
            batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.PRIVATE) 
            alias_sets = get_alias_sets(batch_temps_by_size)
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

        #knl = decouple_domain(knl, knl.inames, frozenset())
        #return b_tunit.with_kernel(knl)
        return tunit.with_kernel(knl)



    b_tunit = linearize_batches(b_tunit, batches)

    print(b_tunit.default_entrypoint)
    #exit()
    #kern = b_tunit.copy(target=lp.CTarget())
    kern = b_tunit.copy(target=lp.OpenCLTarget())
    code = lp.generate_code_v2(kern).device_code()
    print(code)
    exit()
    session = profiler.stop()
    #print(profiler.output_flame())

    #exit()
    #profiler.print(show_all=True)

    #import pyinstrument_flame
    ##renderer = pyinstrument_flame.FlameGraphRenderer(title="Codegen profile", flamechart=True)
    #print(profiler.output(renderer))

    # Requires this branch of pyinstrument
    # https://github.com/cpennington/pyinstrument/tree/flame-chart
    from pyinstrument.renderers import FlameRenderer as Renderer
    renderer = Renderer()#show_all=True)    

    #output = profiler.output(renderer)
    extension = "html"
    #from bs4 import UnicodeDammit
    #svg = UnicodeDammit(svg).unicode_markup
   
    of = open("pyinstrument_output." + extension, "wt")
    #of.write(output)
    of.write(renderer.render(profiler.starting_frame()))
    of.close()
    #print(svg) 

    exit()

    return b_tunit

    #return tunit.with_kernel(knl)

# Maybe add this as an option to lp.add_prefetch if it works
@qprofile
def prefetch_and_project(tunit, argname, prefetch_str, **kwargs):

    prefetch_strings = set(prefetch_str.split(","))

    #print(tunit)
    #print("HERE")
    #exit()   
 
    #before_inames = tunit.default_entrypoint.inames.copy()
    #knl = decouple_domain(tunit.default_entrypoint, prefetch_strings, before_inames.keys())
    #tunit = tunit.with_kernel(knl)
 
    #exit() 
    #return tunit.with_kernel(knl)
    # batch_einsums currently assumes it happens after prefetching,
    # but prefetching and adding the batches will be slow
    # with that many loops. Either the prefetching needs to 
    # occur in the same loops or we need 

    # Figure out the new loops the argname appears in and map the old
    # loops to the new loops
    #print(tunit)
    #knl = decouple_domain(tunit.default_entrypoint, tunit.default_entrypoint.inames, frozenset())
    #tunit = tunit.with_kernel(knl) 

    if False:
        new_inames = [] # Gather all of the new inames in this list
        old_inames = []
        for instr in tunit.default_entrypoint.instructions:
            if argname in instr.dependency_names():
                instr_inames = instr.within_inames
                for new_iname in instr_inames:
                    print(new_iname)
                    old_iname, suffix = new_iname.rsplit("_",1)
                    if old_iname in prefetch_strings:
                        new_inames.append(new_iname)
                        old_inames.append(old_iname)
        #exit()

        new_prefetch_strings = (set(prefetch_strings) - set(old_inames)) | set(new_inames)
        print(new_prefetch_strings)
        print(prefetch_strings)
        #exit()            
        #print(new_inames)
        #print(prefetch_strings)
        # If this is the case then we can just use the new_inames.
        # This implies the variable is only prefetched in a single block
        #assert len(new_inames) == len(prefetch_strings)
        #prefetch_strings = new_inames
        assert len(new_prefetch_strings) == len(prefetch_strings)
        prefetch_strings = new_prefetch_strings
        
    #prefetch_strings = [entry + suffix for entry in prefetch_strings]

    before_inames = tunit.default_entrypoint.inames.copy()
    ts = time.time()
    tunit_new = lp.add_prefetch(tunit, argname, prefetch_strings, **kwargs)
    te = time.time()
    prefetch_time = te - ts
    #tunit = lp.add_prefetch(tunit, argname, prefetch_strings, **kwargs)
    after_inames = tunit_new.default_entrypoint.inames.copy()
    new_inames = set(after_inames.keys()) - set(before_inames.keys())
 
    parent_inames = set(prefetch_strings) | new_inames

    ## Will need to linearize (enforce ordering of, prefetch rules)

    
    ts = time.time()
    # Re-enable this when done testing
    knl = decouple_domain(tunit_new.default_entrypoint, tunit_new.default_entrypoint.inames, frozenset())

    #knl = decouple_domain(tunit_new.default_entrypoint, new_inames | prefetch_strings, frozenset())
    te = time.time()
    decouple_time = te-ts 
    print(prefetch_time, decouple_time)
    #knl = decouple_domain(tunit.default_entrypoint, new_inames, parent_inames)
    #tunit = tunit_new

    #print(tunit)
    #exit()

    tunit = tunit.with_kernel(knl) 
    return tunit

# Assumes there are no internal dependencies between einsums
# Assumes all instructions are einsums
def decompose_batched_einsum_kernel(tunit):
    domain_list = get_domain_list(tunit)

    # Collect the iname tags to reapply to decomposed kernels
    iname_to_tag = []
    for iname, iname_obj in tunit.default_entrypoint.inames.items():
        for tag in iname_obj.tags:
            iname_to_tag.append((iname, tag,))

    print("Incoming tunit")
    print(tunit)
    #exit()

    subkernels = []
    for i, insn in enumerate(tunit.default_entrypoint.instructions):
        print(insn)
        assert len(insn.tags_of_type(EinsumTag)) > 0

        # Get domain(s)
        within_inames = insn.within_inames
        domains = [domain for inames_set, domain in domain_list if within_inames <= inames_set]
        active_vars = insn.dependency_names()
        new_args = [entry for entry in tunit.default_entrypoint.args if entry.name in active_vars]
        temp_args = [entry for entry in tunit.default_entrypoint.temporary_variables.keys() if entry.name in active_vars]

        new_temp_args = []
        for temp in temp_args:
            if temp.address_space == lp.AddressSpace.GLOBAL:
                arg = lp.GlobalArg(temp.name, dtype=temp.dtype, 
                        shape=temp.shape,
                        dim_tags=temp.dim_tags, offset=temp.offset, 
                        dim_names=temp.dim_names,
                        alignment=temp.alignment, tags=temp.tags) 
                        #Any others needed?
                new_args.append(arg)
            else:
                new_temp_args.append(temp)

        new_args += new_temp_args

        name = tunit.default_entrypoint.name + f"_{i}"

        knl = lp.make_kernel(domains, [insn], kernel_data=new_args, name=name)
        #knl = tunit.with_kernel(tunit.default_entrypoint.copy(domains=domains, instructions=[insn], args=new_args, name=name))
        knl = lp.tag_inames(knl, iname_to_tag, ignore_nonexistent=True)
        knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
        print("Output tunit")
        subkernels.append(knl)

        #kern = knl.default_entrypoint.copy(target=lp.CTarget())
        #kern = knl.default_entrypoint.copy(target=lp.OpenCLTarget())
        #kern = lp.add_inames_for_unused_hw_axes(kern)
        #print(kern)
        #code = lp.generate_code_v2(kern).device_code()
        #print(code)
    #exit()

    return subkernels
        

def recompose_batched_einsum_kernel(orig_tunit, subkernels):
    import islpy as isl    

    insns = []
    #assert len(orig_tunit.default_entrypoint.domains) == 1
    #orig_domains = orig_tunit.default_entrypoint.domains[0]
    #print(orig_domains)
    #exit()
    #domains = set()
    #domains = orig_domains
    #print(orig_domains.space)
    #domains = subkernels[0].default_entrypoint.domains[0]
    #domains = list(subkernels[0].default_entrypoint.domains)
    #print(domains.space)
    #print(domains.space)
    var_names = set()
    constraints = []
    args = set()
    temp_args = []
    iname_to_tag = set()

    merged_domain = orig_tunit.default_entrypoint.domains[0]
    temp_args = {}
    for subkernel in subkernels:
        #print("HERE")
        insns = insns + subkernel.default_entrypoint.instructions
        #domains |= set([domain for inames_set, domain in get_domain_list(subkernel)])
        assert len(subkernel.default_entrypoint.domains) == 1
        sk_domains = subkernel.default_entrypoint.domains[0]
        print("Kernel domains", subkernel.default_entrypoint.inames)
        var_names |= set(sk_domains.get_var_dict().keys())
        args |= set(subkernel.default_entrypoint.args)
        #temp_args += subkernel.default_entrypoint.temporary_variables.values()
        temp_args |= subkernel.default_entrypoint.temporary_variables
        #print("Constraints")
        #print(sk_domains.get_constraints())
        constraints += sk_domains.get_constraints()

        for iname, iname_obj in subkernel.default_entrypoint.inames.items():
            for tag in iname_obj.tags:
                iname_to_tag |= set([(iname, tag,)])

        # This is inefficient
        #merged_domain, _ = isl.align_two(merged_domain, sk_domains)

        #print(sk_domains)

        #print("Strides")
        #print(
        #print(subkernel.default_entrypoint.domains[0].project_out_all_params())
        #diff = subkernel.default_entrypoint.domains[0].subtract(orig_domains)
        #print(diff)
        #print(subkernel.default_entrypoint.domains[0].to_list())
        #domains = subkernel.default_entrypoint.domains[0].intersect(domains)
        #for entry in s:
        #print(entry)
        #print(subkernel.default_entrypoint.domains[0].space)
        #domains = domains.union(subkernel.default_entrypoint.domains[0])
        #print(domains.space)
        #domains = domains + list(subkernel.default_entrypoint.domains)
    #print(domains) 
    #print(var_names)
    #print(constraints)
    #print()
    #for constraint in constraints:
    #    print(constraint)
    #    print()
    #exit()

    #print(orig_tunit.default_entrypoint.domains)
    #print(domains)
    #exit()
    #print(domains)
    #print(domains.space)
    #exit()
    
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=var_names)
    domain = isl.BasicSet.universe(space)
    new_constraints = set()
    for constraint in constraints:
        #print()
        #print(constraint)
        coefficients = constraint.get_coefficients_by_name()
        #print(coefficients)
        #print(coefficients)
        if constraint.is_equality():
            new_constraint = isl.Constraint.eq_from_names(space, coefficients=coefficients)
        else:
            new_constraint = isl.Constraint.ineq_from_names(space, coefficients=coefficients)
        new_constraints |= set([new_constraint])
        #print("HERE")
        #print(constraint)
        #print(constraint.get_aff())
        #domain = domain.add_constraint(constraint)
        #print(domain)
    #exit()
    domain = domain.add_constraints(new_constraints)
    #print(domain)
    #exit()
    #print(new_constraints)
    #exit()
    
    # Probably want to turn this off
    #domain = merged_domain
    #knl = lp.make_kernel([domain], insns, 
    #        kernel_data=list(args) + temp_args, 
    #        name=orig_tunit.default_entrypoint.name)
    knl = orig_tunit.with_kernel(orig_tunit.default_entrypoint.copy(domains=[domain], instructions=insns, args=list(args), temporary_variables=temp_args))

    knl = lp.tag_inames(knl, list(iname_to_tag), ignore_nonexistent=False)
    knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))

    # There is still something wrong with this. Sometimes one
    # of the access bounds is dropped.

    #print(knl)
    #print("After make kernel")
    #exit()

    if False:
        knl = lp.add_inames_for_unused_hw_axes(knl)
        #print(knl)
        #print("After add inames")

        #kern = knl.default_entrypoint.copy(target=lp.CTarget())
        kern = knl.default_entrypoint.copy(target=lp.OpenCLTarget())
        #print(knl)
        #print("After add target")

        print("PRINTING GENERATED CODE")
        code = lp.generate_code_v2(kern).device_code()
        print(code)
        print(kern)

        exit()

    return knl

def decompose_and_prefetch(tunit, prefetches, **kwargs):

    """
    print("==================TUNIT BEFORE DECOMPOSITION================")
    knl = tunit.default_entrypoint.copy(target=lp.CTarget())
    knl = lp.add_inames_for_unused_hw_axes(knl)
    print(tunit)
    code = lp.generate_code_v2(knl).device_code()
    print(code)
    
    print("=================DECOMPOSED TUNITS=====================")
    """
    subkernels = decompose_batched_einsum_kernel(tunit)
    output_subkernels = []
    # O(n^2) complexity. Could probably be improved.
    for subkernel in subkernels:
        for prefetch in prefetches:
            arg = prefetch[1][0]
            # Should this be restricted to read args only?
            kernel_args = [kernel_arg.name for kernel_arg in subkernel.default_entrypoint.args]
            if arg in kernel_args:
                subkernel = lp.add_prefetch(subkernel, *prefetch[1], **dict(prefetch[2]))
        output_subkernels.append(subkernel)
        """
        kern = subkernel.default_entrypoint.copy(target=lp.CTarget())
        #kern = subkernel.default_entrypoint.copy(target=lp.OpenCLTarget())
        kern = lp.add_inames_for_unused_hw_axes(kern)
        print("==============DECOMPOSED+PREFETCHED TUNIT=================")
        print(subkernel)
        code = lp.generate_code_v2(kern).device_code()
        print(code)
        """

    # Then recompose the tunit
    #print("====================RECOMPOSED TUNIT========================")
    recomposed = recompose_batched_einsum_kernel(tunit, output_subkernels)

    """
    kern = recomposed.default_entrypoint.copy(target=lp.CTarget())
    kern = lp.add_inames_for_unused_hw_axes(kern)
    print(kern)
    code = lp.generate_code_v2(kern).device_code()
    print(code)
    exit()
    """

    return recomposed

def apply_transformation_list(tunit, transformations):
    # Could just construct a string for the function handle and retrieve the function from that
    function_mapping = {"split_iname": lp.split_iname,
                        "add_prefetch": decompose_and_prefetch,
                        #"add_prefetch": prefetch_and_project,
                        #"add_prefetch": lp.add_prefetch,
                        "prioritize_loops": lp.prioritize_loops,
                        "rename_iname": lp.rename_iname,
                        "tag_array_axes": lp.tag_array_axes,
                        "tag_inames": lp.tag_inames,
                        "batch_einsums": batch_einsums,
                        "add_inames_for_unused_hw_axes": lp.add_inames_for_unused_hw_axes}

    # Maybe add some logic to add slabs=(0,0) if n_elem % k_inner_outer == 0
    # Maybe can do this based on tranformation name, loop variable, and loop variable
    # bounds
    #print("KERNEL BEFORE TRANSFORMATION")
    #print(knl.default_entrypoint)
    print("BEGINNING TRANSFORMATION")
    start = time.time() 

    print(transformations)
    add_prefetches = [t for t in transformations if t[0] == "add_prefetch"]
    #exit()
    # Breaks prefetching
    #for entry in knl.default_entrypoint.inames:
    #    knl = knl.with_kernel(decouple_domain(knl.default_entrypoint, [entry], frozenset()))
    #knl = knl.with_kernel(decouple_domain(knl.default_entrypoint, knl.default_entrypoint.inames, frozenset()))
    
    transformations = list(transformations)
    prefetched=False
    for index, t in enumerate(transformations):

        #print("PRE-TRANSFORMATION CODE")
        #print(knl.default_entrypoint)
        #kern = knl.default_entrypoint.copy(target=lp.OpenCLTarget())
        #code = lp.generate_code_v2(kern).device_code()
        #print(code)

        print(t)
        func = function_mapping[t[0]]
        args = [tunit]
        if len(t) > 1:
            args = args + list(t[1])
        kwargs = dict(t[2]) if len(t) > 2 else {}
        if t[0] == "batch_einsums":
            kwargs["profile"] = True
            tunit = func(*args, **kwargs)
            #exit()
        # Assumes all prefetches are together in the list of transformations
        elif t[0] == "add_prefetch":
            if prefetched == False:
                prefetched=True
                tunit = func(tunit, add_prefetches, profile=True)
                """
                if index == 0 or transformations[index-1][0] != "add_prefetch":
                    # Apply all of the prefetches at once
                    end = index+1
                    while transformations[end][0] == "add_prefetch":
                        end += 1
                    prefetches = transformations[index:end]
                    func(tunit, prefetches, profile=True)

                    #tunit = func(*args, **kwargs)
                    #exit()
                """
        else:
            tunit = func(*args, **kwargs)

    end = time.time()

    print("ENDING TRANSFORMATION:", end - start, "seconds")

    print(tunit.default_entrypoint)
    kern = tunit.default_entrypoint.copy(target=lp.OpenCLTarget())
    code = lp.generate_code_v2(kern).device_code()
    print(code)

    exit()
    return knl
