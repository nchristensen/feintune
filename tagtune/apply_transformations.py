import numpy as np
from pytools import memoize_in, memoize
from meshmode.array_context import EinsumTag
from tagtune.decouple_domain import decouple_domain # decouple_domain can likely be removed.
from tagtune.utils import get_domain_list, get_iname_limits
from frozendict import frozendict
#import pyopencl as cl
#import pyopencl.array
#import pyopencl.clrandom

import loopy as lp
from tagtune.grudge_tags import IsDOFArray, ParameterValue
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

# Should probably rename this to "get_reductions"
def get_einsums(knl):
    einsums = []
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            #print(instr.tags)
            """
            for tag in instr.tags:
                if isinstance(tag, EinsumTag):
                    if isinstance(instr.expression, lp.symbolic.Reduction):
                        einsums.append((instr.within_inames, instr.expression.inames,))
                    else:
                        einsums.append((instr.within_inames, (),))
            """
            #"""
            # Is the above necessary? Can we just look for reductions directly?
            if isinstance(instr.expression, lp.symbolic.Reduction):
                einsums.append((instr.within_inames, instr.expression.inames,))
            else:
                einsums.append((instr.within_inames, (),))
            #"""
    #print(knl.default_entrypoint.name, einsums)
    #exit()
    return einsums


### Can probably delete everything above this line ###

#@memoize
def get_einsum_counts(knl):
    from collections import Counter
    counter = Counter(get_einsums(knl))
    return counter


# Obtain non-reduction and reduction inames 
#@memoize
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

        # Should probably instead check if this is a reduction rather than check for einsum tags
        # see get_einsums
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
@memoize
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
                           
        batch_dict_list.append(frozendict(batch_dict))

    return tuple(batch_dict_list)


#@memoize #Something isn't hashable
def get_alias_sets(batch_dict_list):
    #from itertools import combinations

        arg_to_size = {}
        arg_lists = []
        for batch_dict in batch_dict_list:
            new_list = []
            for size, arg_list in batch_dict.items():
                new_list += arg_list
                for entry in arg_list:
                    arg_to_size[entry] = size
            arg_lists.append(new_list)

        alias_sets = []

        #sizes = set()
        #for batch_dict in batch_dict_list:
        #    sizes |= set(batch_dict.keys())
        


    #for size in sorted(sizes, reverse=True):
    #    arg_lists = []
    #    for i, batch_dict in enumerate(batch_dict_list):
    #        if size in batch_dict:
    #            arg_lists.append(sorted(batch_dict[size]))
    #        else:
    #            arg_lists.append([])


        #max_len = 0
        #for l in arg_lists:
        #    max_len = max(len(l), max_len)

        all_arg_list = set()
        arg_count = 0
        for arg_list in arg_lists:
            arg_count += len(set(arg_list))
            all_arg_list |= set(arg_list)
        # Arange the args to they go from large to small.
        all_arg_list = sorted(all_arg_list, reverse=True, key=lambda arg: arg_to_size[arg])

        aligned_args = np.empty((len(batch_dict_list), len(all_arg_list)), dtype=object)
        aligned_args[:,:] = None

        for row, arg_list in enumerate(arg_lists):
            for arg in arg_list:
                col = all_arg_list.index(arg)
                aligned_args[row][col] = arg

        # Condense the arguments into fewer columns.
        # This is a fairly greedy approach. There are
        # Likely more optimal approaches. Seems to be a bin filling problem.
        col_ind_array = np.arange(0, aligned_args.shape[0])
        for col1 in range(0, len(all_arg_list)):
            for col2 in range(len(all_arg_list)-1, col1, -1):
                ind1 = col_ind_array[aligned_args[:,col1].flatten() != None]
                ind2 = col_ind_array[aligned_args[:,col2].flatten() != None]
                #print(set(ind1) & set(ind2))
                if len(set(ind1) & set(ind2)) == 0:
                    aligned_args[ind2, col1] = aligned_args[ind2, col2]
                    aligned_args[ind2, col2] = None
                    #for index in ind2:
                    #    aligned_args[index,col1] = aligned_args[index,col2]
                    #    aligned_args[index,col2] = None

        #print(aligned_args.shape)
        for col in range(0, len(all_arg_list)):
            if np.all(aligned_args[:,col] == None):
                aligned_args = aligned_args[:,:col+1]
                break
        #print(aligned_args.shape)

        #print(len(aligned_args[aligned_args != None].flatten()), arg_count)
        assert len(aligned_args[aligned_args != None].flatten()) == arg_count
        #print(aligned_args)
        #for entry in aligned_args:
        #    print(aligned_args)
        #exit()
    
        """
        max_len = np.max([len(l) for l in arg_lists])
        for l in arg_lists:
            l += [None]*(max_len - len(l)) # Pad with None so can slice columns


            #while len(l) < max_len:
            #    l.append(None) # Pad with None so can slice columns

        # This needs to be a bit more robust, we want a variable to alias with itself.
        # We also can't allow two tempories to alias if they occur in the same block (row)

        # Permute so if a value is found in the current row it isn't found in the current column
        # except if the value is self
        # Permute so self is in current column as much as possible
        # For now just assert to verify this is not the case, don't attempt to fix

        # We aren't assured swapping entries will suffice to arrange the tuples.
        # Consider (a,b),(b,c),(c,a).

        

        arg_array = np.array(arg_lists)
        print(arg_array)

        moved = []

        # See if any two batches share any temporaries
        set_combo_iterator = combinations(arg_sets,2)
        for (row1, set1), (row2, set2) in set_combo_iterator:
            assert row1 < row2
            intersection = set1 & set2
            #print(intersection)
            for temporary in intersection:
                # Find the column indices of the shared temporaries
                col1 = list(arg_array[row1,:]).index(temporary)
                col2 = list(arg_array[row2,:]).index(temporary)

                moved.append(temporary)

                if col1 != col2: # Already in the same column

                    a_dict = {}
                    def align(d, tup):
                        if tup[0] is None and tup[1] is None:
                            pass
                        elif tup[0] is None and tup[1] is not None:
                            if tup[1] not in d:
                                d[tup[1]] = [True, set()]
                        elif tup[1] is None and tup[0] is not None:
                            if tup[0] not in d:
                                d[tup[0]] = [True, set()]
                        elif tup[0] in d and tup[1] in d:
                            d[tup[0]][1] |= set([tup[1]])
                            d[tup[1]][1] |= set([tup[0]])

                            if d[tup[0]][0] == d[tup[1]][0]:
                                visited = set([tup[0]])
                                to_visit = set([tup[1]])
                                while len(to_visit) > 0:
                                    entry = to_visit.pop()
                                    visited |= set([entry])
                                    d[entry][0] = not d[entry][0]
                                    for item in d[entry][1]:
                                        if item not in visited:
                                            to_visit.append(item)

                        elif tup[0] in d:
                            d[tup[0]][1] |= set([tup[1]])
                            d[tup[1]] = [not d[tup[0]][0], set([tup[0]])]
                        elif tup[1] in d:
                            d[tup[1]][1] |= set([tup[0]])
                            d[tup[0]] = [not d[tup[1]][0], set([tup[1]])]
                        else:
                            d[tup][0] = [True, set([tup[1]])]
                            d[tup][1] = [False, set([tup[0]])]
                     

                    #print((row1,col1), (row2,col2))

                    # Use row1 as the pivot row and swap the col1 and col2 in
                    # the rest of the rows. Need to apply to all rows
                    # except pivot to avoid undoing any prior pairings
        

                    # Attempting to do this using slicing and boolean
                    # arrays didn't work, so doing this manually.
                    #print(temporary, (row1,col1), (row2, col2))

                    #print("BEFORE")
                    #print(arg_array[:,[col1,col2]])

                    for row in np.arange(1, arg_array.shape[0]):
                        
                        if arg_array[row, col2] in arg_array[:row,col1] and arg_array[row,col1] in arg_array[:row,col2]:
                            

                        elif arg_array[row, col2] in arg_array[:row,col1] or arg_array[row,col1] in arg_array[:row,col2]:

                            holder = arg_array[row, col1]
                            arg_array[row,col1] = arg_array[row,col2]              
                            arg_array[row,col2] = holder
                            #if arg_array[row,col1] is not None:
                            #    assert arg_array[row,col1] != arg_array[row,col2]

                    #print("AFTER")
                    # Check that nothing already aligned came out of alignment
                    subarray = arg_array[:,[col1,col2]]
                    for entry in subarray.flatten():
                        if entry is not None:
                            print(entry)
                            indices = np.argwhere(subarray == entry)
                            if entry in moved:
                                assert indices.shape[0] > 1
                                print(indices)
                                print(subarray)
                                assert np.all(indices[:,1] == indices[0,1])
        

                #arg_array[selected_rows,col1][:] = arg_array[selected_rows,col2][:]
                #arg_array[selected_rows,col2][:] = holder[:]

                #print("AFTER")
                #print(arg_array[selected_rows,col1][:])
                #print(arg_array[selected_rows,col2][:])

                #print(arg_array[row1,:])
                #print(arg_array[row2,:])

    
                #arg_array[selected_rows,col1], arg_array[selected_rows,col2] = copy.deepcopy(arg_array[selected_rows, col2]), copy.deepcopy(arg_array[selected_rows,col1])

                # Check that the re-arrangement was done properly
                #assert arg_array[row1, col1] == arg_array[row2, col1]
                #assert arg_array[row1, col1] != arg_array[row2, col2]
                
                #exit()
    # Check that everything is properly aligned.
    for entry in arg_array.flatten():
        if entry is not None:
            indices = np.argwhere(arg_array == entry)
            if not np.all(indices[:,1] == indices[0,1]):
                print(entry, indices)
            assert np.all(indices[:,1] == indices[0,1])

        """
        #flat_arg_array = arg_array.flatten()
        #nonzero_entries = flat_arg_array[np.flatnonzero(flat_arg_array)]
        #unique_entries = np.unique(nonzero_entries)
        #print(unique_entries)
        #print(nonzero_entries)
        # Should be fixed now so this check can be disabled
        #assert len(unique_entries) == len(nonzero_entries)

        #for col in range(arg_array.shape[1]):
        #    col_set = set(arg_array[:,col].flatten())
        #    for row in range(arg_array.shape[0]):
        #        row_set = set(arg_array[row,:].flatten())
        #        assert col_set & row_set == set([arg_array[row,col]])

        # Should start with the largest sets 
        for col in range(aligned_args.shape[1]):
            alias_sets.append(frozenset(aligned_args[:,col]) - frozenset([None]))
    
        return tuple(alias_sets)


from tagtune.qprofile import qprofile, qinstrument

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
        #for i in range(1, nbatches):
        #    j = i - 1
        #    knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")

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
        # Can perhaps do another decompose -> transform -> recompose for creating the loop nests.
        # Essentially rename the inames and then add each set of inames as a separate domain
        # to the recomposed kernel.
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
            #from loopy.transform.realize_reduction import realize_reduction

            for i in range(1, nbatches):
                j = i - 1
                knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")

            pp_tunit = lp.preprocess_program(tunit.with_kernel(knl)) # Realizes the reductions so we can access the accumulators
            # realize_reduction doesn't fill in lp.auto memory spaces
            #pp_tunit = realize_reduction(tunit.with_kernel(knl), unknown_types_ok=True)
            #knl = b_tunit.default_entrypoint

            # Need to separate by address space. Or maybe just call twice?

            batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.LOCAL) 
            alias_sets = get_alias_sets(batch_temps_by_size)
            for s in alias_sets:
                # Synchronizing for exclusive use make loopy fall back to the slow scheduler
                knl = lp.alias_temporaries(knl, list(s), synchronize_for_exclusive_use=False)

            # Doesn't seem to do anything for OpenCL but for other targets it might do something
            batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.PRIVATE) 
            alias_sets = get_alias_sets(batch_temps_by_size)
            for s in alias_sets:
                knl = lp.alias_temporaries(knl, list(s), synchronize_for_exclusive_use=False)

        if False:

            # Using global barriers helps, but not by much. It still takes minutes
            # to generate the code.

            for i in range(1, nbatches):
                j = i - 1
                knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")
           
            store_insns = ["id:"+instr.id for instr in tunit.default_entrypoint.instructions if "_store" in instr.id]
            fetch_insns = ["id:"+instr.id for instr in tunit.default_entrypoint.instructions if "_fetch" in instr.id]
            print("STORE INSTRUCTIONS")
            print(store_insns)
            print("FETCH INSTRUCTIONS")
            print(fetch_insns)
            #exit()
            for i in range(1,nbatches,1):
                knl = lp.add_barrier(knl, insn_before=store_insns[i-1], insn_after=fetch_insns[i])
            
        
        #print(knl)
        #exit()
        
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
        out_tunit = tunit.with_kernel(knl)
        #return lp.save_and_reload_temporaries(out_tunit)
        return out_tunit


    b_tunit = linearize_batches(b_tunit, batches)

    print(b_tunit.default_entrypoint)
    #exit()
    #kern = b_tunit.copy(target=lp.CTarget())
    #kern = b_tunit.copy(target=lp.OpenCLTarget())
    kern = b_tunit
    start = time.time()
    code = lp.generate_code_v2(kern).device_code()
    end = time.time()
    print(code)
    print("Codegen time:", end-start)
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
@memoize
def decompose_batched_einsum_kernel(tunit):
    domain_list = get_domain_list(tunit)

    # Collect the iname tags to reapply to decomposed kernels
    iname_to_tag = []
    for iname, iname_obj in tunit.default_entrypoint.inames.items():
        for tag in iname_obj.tags:
            iname_to_tag.append((iname, tag,))

    #print("Incoming tunit")
    #print(tunit)
    #exit()

    subkernels = []
    for i, insn in enumerate(tunit.default_entrypoint.instructions):

        # Get domain(s)
        within_inames = insn.within_inames
        domains = [domain for inames_set, domain in domain_list if within_inames <= inames_set]
        #if False:#len(tunit.default_entrypoint.domains) > 1:
        #    from islpy import align_two
        #    domains = tunit.default_entrypoint.domains
        #    new_domains = domains[0]
        #    for i in range(1,len(domains)):
        #        b1, b2 = align_two(new_domains, domains[i])
        #        new_domains = b1 | b2
        #    domains = [new_domains]

        active_vars = insn.dependency_names()
        new_args = [entry for entry in tunit.default_entrypoint.args if entry.name in active_vars]
        temp_args = [entry for entry in tunit.default_entrypoint.temporary_variables.values() if entry.name in active_vars]

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

    return tuple(subkernels)
        
# Maybe this should be added to loopy
from loopy.translation_unit import for_each_kernel

@for_each_kernel
def merge_prefetch_inames(knl, prefetch_inames):

    prefetch_inames = set(prefetch_inames)
    prefetch_instrs = [instr for instr in knl.instructions if "fetch_rule" in instr.id]
    iname_limits = get_iname_limits(knl)
    d = {}

    #new_inames = (set( 
    # I'm trying to reduce the number of loop domains here by combining prefetch loops.
    #prefetch_str = prefetch[1][1]
    #key = (prefetch_str, added_iname_limits,)

    #subkernel = lp.rename_iname(subkernel, remove, keep, existing_ok=True, preserve_tags=False)
    
    """
    # Doesn't work. Comparing by prefetch_str is not sufficient. Need to look at domain bounds as well.
    #print(key)
    if key in prefetch_iname_dict:
        existing_fetch_inames = d[key]
        #from tagtune.utils import get_domain_list
        #dl = dict(get_domain_list(subkernel))
        #print("KEYS", dl.keys())
        #for entry in dl.keys():
        #    print(entry)
        #print("EXISTING", frozenset(existing_fetch_inames))
        #print("ADDED", frozenset(added_inames))
        #print("DOMAINS", subkernel.default_entrypoint.domains)
        #assert frozenset(existing_fetch_inames) in dl
        #assert frozenset(added_inames) in dl
        #new_inames = (set(subkernel.default_entrypoint.inames.keys()) - orig_inames) - existing_fetch_inames
        #print(subkernel)
        for remove, keep in zip(sorted(added_inames,reverse=True), sorted(existing_fetch_inames,reverse=True)):
            print("RENAMING FETCH INAMES:", remove, "->", keep)
            subkernel = lp.rename_iname(subkernel, remove, keep, existing_ok=True,
                                        preserve_tags=False, raise_on_domain_mismatch=False)
        #exit()
    else:
        #prefetch_inames = (set(subkernel.default_entrypoint.inames.keys()) - orig_inames) - set().union(*(prefetch_iname_dict.values()))
        prefetch_iname_dict[key] = added_inames
        print("ADDING KEY TO DICTIONARY")
        print(prefetch_iname_dict)
    """

    
    for prefetch in prefetch_instrs:
        print("PREFETCH INAMES", prefetch_inames)
        outer_inames = prefetch.within_inames - prefetch_inames
        print("OUTER INAMES", outer_inames)
        inner_inames = sorted(prefetch.within_inames - outer_inames, key=lambda iname: str(prefetch).index(iname))
        print("INNER INAMES", inner_inames)
        prefetch_iname_limits = tuple(sorted([iname_limits[prefetch_iname] for prefetch_iname in inner_inames]))
        key = (tuple(sorted(outer_inames)), prefetch_iname_limits)         
        if key in d:
            existing_fetch_inames = d[key]
            for remove, keep in zip(inner_inames, existing_fetch_inames):
                print("RENAMING BATCH FETCH INAMES:", remove, "->", keep)
                if remove != keep:
                    knl = lp.rename_iname(knl, remove, keep, existing_ok=True, preserve_tags=False, raise_on_domain_mismatch=False)
        else:
            print("ADDING", key, "TO PREFETCH DICT")
            d[key] = inner_inames
    
    return knl


def recompose_batched_einsum_kernel(orig_tunit, subkernels, batch_size=0):
    import islpy as isl    

    if batch_size == 0:
        batch_size = len(subkernels)
    batch_size = min(batch_size, len(subkernels))
    nbatches = np.int32(np.ceil(len(subkernels) / batch_size))

    insns = []
    args = set()
    temp_args = {}
    iname_to_tag = set()
    domains = []
    single_batch_kernel = None

    # Assuming there are no internal dependencies, then we can reduce the number of prefetching
    # operations by putting instructions that do the same prefetches in the same batches
    """ 
    sk_fetch_rules = []
    for subkernel in subkernels:
        fetch_rules = {instr.id for instr in subkernel.instructions if "_fetch" in instr.id}
        sk_fetch_rules.append(fetch_rules)
    from itertools import combinations
    nintersections = 0
    for (sk_id_1, set1), (sk_id_2, set2) in combinations(sk_fetch_rules, 2):
        intersection = set1 & set2
        nintersections += len(intersection)

    ## Better idea, find the two sets the intersect the most, put the corresponding instructions in a batch.
    ## Then recompute the intersections with the remaining instructions and put the sets that intersect the
    ## most in a different batch if possible, otherwise in the set with the least intersections.
    ## Then fill out the sets with the sets with the intersectionless einsums. Or do it in reverse
    ## and fill out the sets with intersectionless einsums, leaving space for pairs of intersecting einsums.
    ## This isn't quite right though. What if an einsum shares 4 single prefetches with other einsums but.
    ## 3 prefetches with one single einsum.
    ## There also may be a trade off between reducing the number of prefetches vs. reducing the amount of local
    ## memory used (although this is unlikely).
    ## Probably should prioritize reducing the local memory used if possible, then attempt to reduce
    ## the number of prefetches.
    """

    # Assemble the sub-batches
    print("ASSEMBLING SUB-BATCHES")

    prefetch_inames = []
    for batch in range(nbatches):
        var_names = set()
        constraints = []
        for subkernel in subkernels[batch*batch_size:(batch+1)*batch_size]:

            sk = subkernel.default_entrypoint

            for old_iname in sk.inames.keys():
                if old_iname not in orig_tunit.default_entrypoint.inames:
                    prefetch_inames.append(old_iname + f"_b{batch}")

                sk = lp.rename_iname(sk, old_iname, old_iname + f"_b{batch}", existing_ok=False, preserve_tags=True)
                sk = lp.remove_unused_inames(sk)
                # For some reason it doesn't always remove inames with constant constraints
                if old_iname in sk.inames:
                    new_domains = []
                    for domain in sk.domains:
                        dt, idx = domain.get_var_dict()[old_iname]
                        new_domains.append(domain.remove_dims(dt, idx, 1))
                    ins = [instr.copy(within_inames=instr.within_inames - frozenset([old_iname])) for instr in sk.instructions]
                    sk = sk.copy(domains=new_domains, instructions=ins)
                    # If this doesn't work, will need to go through and manually remove from each iname
                    
                    #sk = lp.untag_inames(sk, old_iname, loopy.kernel.data.InameImplementationTag)
                print(sk)
                assert old_iname not in sk.inames 

            insn_mappings = {instr.id: [f"batch_{batch}_" + instr.id] for instr in sk.instructions}
            sk = lp.replace_instruction_ids(sk, insn_mappings)
            
            """
            # For some reason it won't otherwise consistently delete the unused inames.
            if any([iname.endswith(f"_b{batch}") for iname in sk.inames.keys()]):
                unused = [iname for iname in sk.inames.keys() if not iname.endswith(f"_b{batch}")]
                from tagtune.decouple_domain import decouple_domain
                if len(unused) > 0:
                    sk = decouple_domain(sk, unused, frozenset())
                sk = lp.remove_unused_inames(sk, inames=unused)
            """

            # This seems to make some inames non-removable
            sk = lp.add_inames_for_unused_hw_axes(sk)
            #sk = lp.remove_unused_inames(sk, inames=["idof_ensm2_0_outer"])
            insns = insns + sk.instructions
            #print(sk)

            #assert len(sk.domains) == 1
            for sk_domains in sk.domains:
                #sk_domains = sk.domains[0]
                var_names |= set(sk_domains.get_var_dict().keys())
                args |= set(sk.args)
                temp_args |= sk.temporary_variables
                constraints += sk_domains.get_constraints()

                for iname, iname_obj in sk.inames.items():
                    for tag in iname_obj.tags:
                        iname_to_tag |= set([(iname, tag,)])

        # Eliminate redundant (prefetch) instructions:
        insns = list(set(insns))
        
        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=var_names)
        domain = isl.BasicSet.universe(space)
        new_constraints = set()
        for constraint in constraints:
            coefficients = constraint.get_coefficients_by_name()
            if constraint.is_equality():
                new_constraint = isl.Constraint.eq_from_names(space, coefficients=coefficients)
            else:
                new_constraint = isl.Constraint.ineq_from_names(space, coefficients=coefficients)
            new_constraints |= set([new_constraint])

        domain = domain.add_constraints(new_constraints)
        domains.append(domain)

        # TODO: Merge prefetch domains in the same batch where possible.
        # DONE
        if batch == 0:
            # Create a single batch knl, which may be faster to tune with
            single_batch_knl = orig_tunit.with_kernel(orig_tunit.default_entrypoint.copy(domains=domains, 
                                    instructions=insns, args=list(args), temporary_variables=temp_args))
            single_batch_knl = lp.tag_inames(single_batch_knl, list(iname_to_tag), ignore_nonexistent=True)
            single_batch_knl = lp.set_options(single_batch_knl, lp.Options(no_numpy=True, return_dict=True))
            #single_batch_knl = lp.add_inames_for_unused_hw_axes(single_batch_knl)

    knl = lp.make_kernel(domains, insns, 
            kernel_data=list(args) + list(temp_args.values()), 
            name=orig_tunit.default_entrypoint.name)

    # Avoids some weird errors with make_kernel
    #knl = orig_tunit.with_kernel(orig_tunit.default_entrypoint.copy(domains=domains, 
    #        instructions=insns, args=list(args), temporary_variables=temp_args))
    knl = lp.tag_inames(knl, list(iname_to_tag), ignore_nonexistent=False)
    knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
    knl = merge_prefetch_inames(knl, prefetch_inames)
    #knl = lp.add_inames_for_unused_hw_axes(knl)

    if True:
        if nbatches > 1: #Pointless of alias temporaries if there is a single batch
            print("ALIASING TEMPORARIES")
            for i in range(1, nbatches):
                j = i - 1
                knl = lp.add_dependency(knl, f"id:batch_{i}_*", f"id:batch_{j}_*")

            if True:

                #from loopy.transform.realize_reduction import realize_reduction
                print("PREPROCESSING PROGRAM")
                pp_tunit = lp.preprocess_program(knl) # Realizes the reductions so we can access the accumulators and fill in lp.auto memory spaces
                # realize_reduction doesn't fill in lp.auto memory spaces
                #pp_tunit = realize_reduction(tunit.with_kernel(knl), unknown_types_ok=True)
                #knl = b_tunit.default_entrypoint
                print("LOCAL: GETTING BATCH TEMPORARIES BY SIZE")
                batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.LOCAL) 
                print("LOCAL: GETTING ALIAS SETS")
                alias_sets = get_alias_sets(batch_temps_by_size)
                print("LOCAL: CALLING ALIAS TEMPORARIES")
                for s in alias_sets:
                    print(sorted(s))
                    # Synchronizing for exclusive use make loopy fall back to the slow scheduler
                    # ILP also causes use of the slow scheduler, but oh well.
                    knl = lp.alias_temporaries(knl, list(s), synchronize_for_exclusive_use=False)

                # Doesn't seem to do anything for OpenCL but for other targets it might do something
                print("PRIVATE: GETTING BATCH TEMPORARIES BY SIZE")
                batch_temps_by_size = get_batch_temporaries_by_size(pp_tunit, nbatches, lp.AddressSpace.PRIVATE) 
                print("PRIVATE: GETTING ALIAS SETS")
                alias_sets = get_alias_sets(batch_temps_by_size)
                print("PRIVATE: CALLING ALIAS SETS")
                for s in alias_sets:
                    knl = lp.alias_temporaries(knl, list(s), synchronize_for_exclusive_use=False)

        if False:

            # Using global barriers helps, but not by much. It still takes minutes
            # to generate the code.

            # Seems to conflict with add_inames_for_unused_hw_axes.

            store_insns = ["id:"+instr.id for instr in knl.default_entrypoint.instructions if "_store" in instr.id]
            fetch_insns = ["id:"+instr.id for instr in knl.default_entrypoint.instructions if "_fetch" in instr.id]
            print("STORE INSTRUCTIONS")
            print(store_insns)
            print("FETCH INSTRUCTIONS")
            print(fetch_insns)
            #exit()
            for i in range(1,nbatches,1):
                knl = lp.add_barrier(knl, insn_before=store_insns[i-1], insn_after=fetch_insns[i])
            
   

    if False:
        knl = lp.add_inames_for_unused_hw_axes(knl)
        #kern = knl.default_entrypoint.copy(target=lp.CTarget())
        kern = knl.default_entrypoint.copy(target=lp.OpenCLTarget())

        print("PRINTING GENERATED CODE")
        print(kern)
        start = time.time()
        code = lp.generate_code_v2(kern).device_code()
        print(code)
        end=time.time()
        print(end-start)


    #print("SINGLE BATCH KERNEL")
    #print(single_batch_knl)
    #print(knl)
    #exit()

    return knl, single_batch_knl

# Test code. This should be handled in loopy most likely. Or maybe in pytato
"""
def prune_conditionals(instr):

    class MyPrunerMapper(lp.symbolic.UncachedIdentityMapper):

        def map_if(self, expr, *args, seen_conditions=frozenset()):
            print("SEEN CONDITIONS")
            print(str(expr.condition), seen_conditions, str(expr.condition) in seen_conditions)
            print("CONDITIONAL")
            print("CONDITION", expr.condition) 
            print("THEN", expr.then)
            print("ELSE", expr.else_)
            
            if str(expr.condition) in seen_conditions:
                result = self.rec(expr.else_, *args, seen_conditions=seen_conditions)
                print(result)
                exit()
                return result
            else:
                seen_conditions |= frozenset([str(expr.condition)])
                then = self.rec(expr.then, *args, seen_conditions=seen_conditions)
                else_ = self.rec(expr.else_, *args, seen_conditions=seen_conditions)
                from pymbolic.primitives import If
                ans = If(expr.condition, then, else_)
                return ans

    if isinstance(instr, lp.Assignment):
        prune = MyPrunerMapper()
        expr_pruned = prune.rec(instr.expression)
        print("DONE PRUNING")

        if expr_pruned != instr.expression:
            print("PRUNED INSTRUCTION")
            print(instr_pruned)
            print("NONPRUNED INSTRUCTION")
            print(instr)
            exit()
        # Re-write instruction with new expression.
    return instr
"""

@qprofile
def decompose_and_prefetch(tunit, prefetches, batch_size=0, **kwargs):

    """
    print("==================TUNIT BEFORE DECOMPOSITION================")
    knl = tunit.default_entrypoint.copy(target=lp.CTarget())
    knl = lp.add_inames_for_unused_hw_axes(knl)
    print(tunit)
    code = lp.generate_code_v2(knl).device_code()
    print(code)
    
    print("=================DECOMPOSED TUNITS=====================")
    """
      
    if False:#len(get_einsums(tunit)) == 1:
        # Kernel only has one einsum. No need to decompose it.
        # Merge the domains if there is more than one. Inhibits prefetching
        if len(tunit.default_entrypoint.domains) > 1:
            from islpy import align_two
            domains = tunit.default_entrypoint.domains
            new_domains = domains[0]
            for i in range(1,len(domains)):
                b1, b2 = align_two(new_domains, domains[i])
                new_domains = b1 | b2
            #print(domains)
            #print(new_domains)
            #exit()
            domains = [new_domains]
            tunit = tunit.with_kernel(tunit.default_entrypoint.copy(domains=domains))

        subkernels = [tunit]
    else:

        print("BEGINNING DECOMPOSION")
        subkernels = decompose_batched_einsum_kernel(tunit)
        print("ENDING DECOMPOSITION")
    output_subkernels = []

    for subkernel in subkernels:
        kernel_args = {kernel_arg.name for kernel_arg in subkernel.default_entrypoint.args}
        subknl_prefetches = [prefetch for prefetch in prefetches if prefetch[1][0] in kernel_args]

        # Pointless to prefetch if a single einsum use a huge number of DOF arrays. Very little
        # data would be in local memory and it takes forever to generate the kernel because the loop domains
        # become so large. Arbitrarily setting the cutoff to 10.
        cutoff = np.inf#10#np.inf

        #orig_inames = set(subkernel.default_entrypoint.inames.keys())
        prefetch_iname_dict = {}
        
        #if len(subknl_prefetches) > 1:
        #    print(subknl_prefetches)
        #    print(prefetches)
        #    print("EINSUM has more than one prefetch")
        #    exit()

        #print(subkernel)

        if len(subknl_prefetches) <= cutoff:
            # Should this be restricted to read args only? How should deeply nested if-statements be handled?
            #kernel_args = [kernel_arg.name for kernel_arg in subkernel.default_entrypoint.args]
            for prefetch in subknl_prefetches:
                #print(prefetch)
                before_inames = set(subkernel.default_entrypoint.inames.keys())
                subkernel = lp.add_prefetch(subkernel, *prefetch[1], **dict(prefetch[2]))
                after_inames = set(subkernel.default_entrypoint.inames.keys())
                added_inames = sorted(after_inames - before_inames)

                #TODO: Switch to merge_prefetch_inames implementation.
                subkernel = merge_prefetch_inames(subkernel, added_inames)
                '''
                iname_limits = get_iname_limits(subkernel.default_entrypoint)
                added_iname_limits = tuple([iname_limits[added_iname] for added_iname in added_inames])

                #new_inames = (set( 
                # I'm trying to reduce the number of loop domains here by combining prefetch loops.
                prefetch_str = prefetch[1][1]
                key = (prefetch_str, added_iname_limits,)

                #subkernel = lp.rename_iname(subkernel, remove, keep, existing_ok=True, preserve_tags=False)
                
                #"""
                # Doesn't work. Comparing by prefetch_str is not sufficient. Need to look at domain bounds as well.
                #print(key)
                if True:
                    if key in prefetch_iname_dict:
                        existing_fetch_inames = prefetch_iname_dict[key]
                        #from tagtune.utils import get_domain_list
                        #dl = dict(get_domain_list(subkernel))
                        #print("KEYS", dl.keys())
                        #for entry in dl.keys():
                        #    print(entry)
                        #print("EXISTING", frozenset(existing_fetch_inames))
                        #print("ADDED", frozenset(added_inames))
                        #print("DOMAINS", subkernel.default_entrypoint.domains)
                        #assert frozenset(existing_fetch_inames) in dl
                        #assert frozenset(added_inames) in dl
                        #new_inames = (set(subkernel.default_entrypoint.inames.keys()) - orig_inames) - existing_fetch_inames
                        #print(subkernel)
                        for remove, keep in zip(sorted(added_inames,reverse=True), sorted(existing_fetch_inames,reverse=True)):
                            print("RENAMING FETCH INAMES:", remove, "->", keep)
                            subkernel = lp.rename_iname(subkernel, remove, keep, existing_ok=True,
                                                        preserve_tags=False, raise_on_domain_mismatch=False)
                        #exit()
                    else:
                        #prefetch_inames = (set(subkernel.default_entrypoint.inames.keys()) - orig_inames) - set().union(*(prefetch_iname_dict.values()))
                        prefetch_iname_dict[key] = added_inames
                        print("ADDING KEY TO DICTIONARY")
                        print(prefetch_iname_dict)
                    #"""
                '''
        else:
            print(f"Prefetching on einsum disabled. Einsum uses more than {cutoff} prefetchable arguments.")

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
    print("BEGINNING RECOMPOSITION")
    recomposed, single_batch_knl = recompose_batched_einsum_kernel(tunit, output_subkernels, batch_size=batch_size)
    print("ENDING RECOMPOSITION")

    """
    kern = recomposed.default_entrypoint.copy(target=lp.CTarget())
    kern = lp.add_inames_for_unused_hw_axes(kern)
    print(kern)
    code = lp.generate_code_v2(kern).device_code()
    print(code)
    exit()
    """

    return recomposed, single_batch_knl

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

    batch_size = 0
    for t in transformations:
        if t[0] == "batch_einsums":
            batch_size = t[1][0]
    #batch_size = 1

    transformations = list(transformations)
    prefetched=False
    sb_tunit = None
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
            # Now handled by add_prefetch
            pass
            #kwargs["profile"] = False
            #tunit = func(*args, **kwargs)
        # Assumes all prefetches are together in the list of transformations
        elif t[0] == "add_prefetch" or t[0] == "batch_einsums":
            # TODO Allow batching without prefetching.
            if prefetched == False: # This needs to be a separate if statement to prevent falling into the final else
                prefetched=True
                tunit, sb_tunit = func(tunit, add_prefetches, batch_size=batch_size, profile=False)
        else:
            tunit = func(*args, **kwargs)

    # Assumes add_prefetch happens last
    # TODO: Only apply this if it is in the transform list.
    #tunit = lp.add_inames_for_unused_hw_axes(tunit)
    #if sb_tunit is not None:
    #    sb_tunit = lp.add_inames_for_unused_hw_axes(sb_tunit)

    end = time.time()

    print("ENDING TRANSFORMATION:", end - start, "seconds")


    print("SINGLE BATCH TUNIT", batch_size)
    print(sb_tunit)

    if False:
        print(sb_tunit.default_entrypoint)
        kern = sb_tunit.default_entrypoint.copy(target=lp.OpenCLTarget())
        start = time.time()
        code = lp.generate_code_v2(kern).device_code()
        end = time.time()
        print(code)

        print("Codegen time:", end-start)
        exit()

    if False:
        print(tunit.default_entrypoint)
        kern = tunit.default_entrypoint.copy(target=lp.OpenCLTarget())
        start = time.time()
        code = lp.generate_code_v2(kern).device_code()
        end = time.time()
        print(code)

        print("Codegen time:", end-start)
        #exit()

    return tunit, sb_tunit