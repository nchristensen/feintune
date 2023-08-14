import numpy as np
import pickle
import loopy as lp
from pytools.tag import Tag
from meshmode.array_context import EinsumTag
import pyopencl as cl
import os
from os.path import exists
from utils import unique_program_id, convert, load_hjson, dump_hjson, get_domain_list, get_indirection_arrays
import hjson
from generators import createConfigSpace
from ytopt_autotuning import ytopt_tuning
from time import time

use_charm=False
if use_charm:
    from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
    from charm4py.pool import PoolScheduler, Pool
    from charm4py.charm import Charm, CharmRemote
    from parallel_autotuning_charm4py_v2 import parallel_autotune
else:
    from parallel_autotuning_mpi4py_v2 import parallel_autotune
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD

from generators import einsum3to2_kernel_tlist_generator_v2#, einsum4to2_kernel_tlist_generator_v2
from run_tests import run_single_param_set_v2
from run_tests import generic_test


# Get the barriers to divide computation into phases
def get_barriers(tunit):
    barriers = [None]
    for instr in tunit.default_entrypoint.instructions:
        if isinstance(instr, lp.BarrierInstruction) and instr.synchronization_kind == "global":
            barriers.append(instr.id)
    #print("Number of global barriers", len(barriers))
    return barriers


# Get the barriers to divide computation into phases
def get_phases(tunit, barriers):

    # Should a phase be an object?
    phase_lists = [{"domains": frozenset(), "within_inames": frozenset(), "instructions": [], "args": frozenset()} for i in range(len(barriers) + 1)]
    phases = dict(zip(barriers, phase_lists))
    for instr in tunit.default_entrypoint.instructions:
        dbarrier = None
        for entry in instr.depends_on:
            if entry in barriers:
                dbarrier = entry
                break

        phases[dbarrier]["instructions"].append(instr)
        phases[dbarrier]["within_inames"] = instr.within_inames | phases[dbarrier]["within_inames"]

    # Replace the text domain names with the actual domain objects
    domain_list = get_domain_list(tunit)
    # Determine the domain objects from the inames
    for dbarrier in barriers:
        
        within_inames = phases[dbarrier]["within_inames"]
        phases[dbarrier]["domains"] = []

        for inames_set, domain in domain_list:
            if within_inames <= inames_set:
                phases[dbarrier]["domains"].append(domain)

        #print(len(phases[dbarrier]["domains"]))

    return phases

# Strip off the dependencies on global barriers and other phases
def strip_unused_dependencies(instructions):
    phase_instruction_ids = [instruction.id for instruction in instructions]

    new_instructions = []
    barrier_dep_count = {}

    # Collect the barrier instructions
    for instruction in instructions:
        if isinstance(instruction, lp.BarrierInstruction):
            barrier_dep_count[instruction.id] = 0            

    for instruction in instructions:
        new_dependencies = []
        for dependency in instruction.depends_on:
            if dependency in phase_instruction_ids:
                new_dependencies.append(dependency)
            if dependency in barrier_dep_count:
                barrier_dep_count[dependency] += 1

        new_instruction = instruction.copy()
        new_instruction.depends_on = frozenset(new_dependencies)
        new_instructions.append(new_instruction)

    # Strip off unused barrier instructions
    new_new_instructions = []
    for instruction in new_instructions:
        if not (isinstance(instruction, lp.BarrierInstruction) and barrier_dep_count[instruction.id] == 0):
            new_new_instructions.append(instruction)

    return new_new_instructions

def assemble_transformed_macrokernel(macrokernel, subkernels):
    # Get the barrier instructions
    barriers = [instr for instr in macrokernel.default_entrypoint.instructions if isinstance(instruction, lp.BarrierInstruction) and instr.synchronization_kind == "global"]

    assert len(subkernels) - 1 == len(barriers)
    new_instructions = subkernels[0].instructions
    domains = subkernels[0].domains
    for barrier, subkernel in zip(barriers, subkernels[1:]):
        print(barrier.depends_on)
        """
        new_instructions.append(barrier)
        domains = domains + subkernel.domains
        for sk_instruction in subkernel.default_entrypoint.instructions:
            sk_instruction.depends_on |= frozenset([barrier.id])
            new_instructions.append(sk_instruction)

    # Also need to copy the domains and inames
    new_macrokernel = macrokernel.default_entrypoint.copy(instructions=new_instructions, domains=domains)
        """
    exit();
    return macrokernel.with_kernel(new_macrokernel)

# FIXME: Needs to look at the csv file instead of the hjson file
# FIXME: Just return a list of transform dictionaries.
def transform_macrokernel(tunit, args, save_path):
    
    subkernels = get_subkernels(tunit, args)
    transformed_subkernels = []
    for sk in subkernels:
        indirection = len(get_indirection_arrays(sk)) > 0
        pid = unique_program_id(sk)
        hjson_file_str = save_path + "/" + pid + ".hjson"
        #if not exists 
            # Tune the subkernel
        hjson = load_hjson(hjson_file_str)
        # Transform ...

        transformed_subkernels.append(transformed_subkernel)


    transformed_tunit = assemble_transformed_macrokernel(tunit, transformed_subkernels)
    return transformed_tunit

# Create a subkernel with the domains and instructions of each cumulative phase
def generate_cumulative_subkernels(tunit, barriers, phases):
    subkernels = []    
    for cur_phase in range(len(barriers)):
        #print(f"BARRIER {barriers[cur_phase]}")
        domains = []
        instructions = []
        for i in range(cur_phase + 1):
            domains += phases[barriers[i]]["domains"]
            #print(domains)
            instructions += phases[barriers[i]]["instructions"]
        instructions = strip_unused_dependencies(instructions)

        active_vars = frozenset()
        for instruction in instructions:
            active_vars |= instruction.dependency_names()

        # Strip off unused args
        new_args = []
        for entry in tunit.default_entrypoint.args:
            if entry.name in active_vars:
                new_args.append(entry)

        for entry in tunit.default_entrypoint.temporary_variables.keys():
            if entry in active_vars:
                new_args.append(tunit.default_entrypoint.temporary_variables[entry])

        name = tunit.default_entrypoint.name + f"_{cur_phase}_cum"
        knl = lp.make_kernel(domains, instructions, kernel_data=new_args, name=name)
        knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
        subkernels.append(knl)
    return subkernels

# Create a subkernel with the domains and instructions of each single phase
def generate_subkernels(tunit, barriers, phases):
    subkernels = []
    for cur_phase in range(len(barriers)):
        #print(f"BARRIER {barriers[cur_phase]}")
        domains = phases[barriers[cur_phase]]["domains"]
        instructions = phases[barriers[cur_phase]]["instructions"]
        instructions = strip_unused_dependencies(instructions)

        active_vars = frozenset()
        for instruction in instructions:
            active_vars |= instruction.dependency_names()

        # Strip off unused args
        new_args = []
        for entry in tunit.default_entrypoint.args:
            if entry.name in active_vars:
                new_args.append(entry)

        temp_args = []
        for entry in tunit.default_entrypoint.temporary_variables.keys():
            if entry in active_vars:
                temp_args.append(tunit.default_entrypoint.temporary_variables[entry])

        # Should also make sure temporaries that are read before they are written
        # are made args instead of temporary args, for now just do for all GlobalArgs.
        new_temp_args = []
        for temp in temp_args:
            if temp.address_space == lp.AddressSpace.GLOBAL:
                """
                # Fails for some reason
                from copy import deepcopy
                name = deepcopy(temp.name)
                tdict = vars(temp)
                del tdict["name"]
                del tdict["read_only"]
                del tdict["base_indices"]
                del tdict["_base_storage_access_may_be_aliasing"]
                del tdict["storage_shape"]
                del tdict["base_storage"]
                del tdict["initializer"]
                #arg = lp.ArrayArg(name, **tdict)
                """

                arg = lp.GlobalArg(temp.name, dtype=temp.dtype, shape=temp.shape,
                        dim_tags=temp.dim_tags, offset=temp.offset, dim_names=temp.dim_names,
                        alignment=temp.alignment, tags=temp.tags) #Any others needed?
                new_args.append(arg)
            else:
                new_temp_args.append(temp)

        new_args += new_temp_args
        name = tunit.default_entrypoint.name + f"_{cur_phase}"
        #print("DOMAINS")
        #print(domains)
        #for domain in domains:
        #    print(domain)
        #for instruction in instructions:
        #    print(instruction)
        knl = lp.make_kernel(domains, instructions, kernel_data=new_args, name=name)
        knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
        #if knl.default_entrypoint.name == "unfiltered_rhs_5" and ("pt_temp_119" in knl.default_entrypoint.args or "pt_temp_119" in knl.default_entrypoint.temporary_variables.values()):
        #    print(knl)
        #    exit()
        subkernels.append(knl)
    return subkernels


from __init__ import get_einsums, get_einsum_counts, get_einsum_types

def dump_subkernels_from_pickled(arg):

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    directory="./pickled_programs/pickled_programs_eighthX_order_4"
    files = os.listdir(directory)
    for num, filename in list(enumerate(sorted(files))):
        print(num, filename)
        f = os.path.join(directory, filename)
        # Skip the massive kernel for now
        if os.path.isfile(f) and filename.startswith("prefeinsum") and (filename.endswith(".pickle") or filename.endswith(".pkl")):
            f = open(f, "rb")
            tunit, args = pickle.load(f)
            f.close()
            sks = get_subkernels(tunit, args)
            if len(sks) == 1:
                print(sks[0][0].default_entrypoint)
                
                if len(get_einsum_types(sks[0][0])) > 0:
                    autotune_parts(sks, queue)
                """
                for sk,csk in sks:
                    print(sk.default_entrypoint())
                for sk,sk in sks:
                    print(get_einsum_types(sk))
                """
    #exit()

def feinsum_autotune(t_unit, queue):
    from loopy.match import ObjTagged
    import feinsum as fnsm
    from functools import reduce
    from meshmode.feinsum_transformations import FEINSUM_TO_TRANSFORMS

    assert all(insn.tags_of_type(EinsumTag)
           for insn in t_unit.default_entrypoint.instructions
           if isinstance(insn, lp.MultiAssignmentBase)
           )

    einsum_tags = reduce(
        frozenset.union,
        (insn.tags_of_type(EinsumTag)
         for insn in t_unit.default_entrypoint.instructions),
        frozenset())
    for ensm_tag in sorted(einsum_tags,
       	    key=lambda x: sorted(x.orig_loop_nest)):
        if reduce(frozenset.union,
            (insn.reduction_inames()
                   for insn in (t_unit.default_entrypoint.instructions)
                   if ensm_tag in insn.tags),
             frozenset()):
            fused_einsum = fnsm.match_einsum(t_unit, ObjTagged(ensm_tag))
        else:
            # elementwise loop
            from meshmode.array_context import _get_elementwise_einsum
            fused_einsum = _get_elementwise_einsum(t_unit, ensm_tag)


        normalized_fused_einsum = fnsm.normalize_einsum(fused_einsum)
        print(normalized_fused_einsum)


# Copied from Meshmode
def apply_feinsum_transformations(t_unit, queue):
    from loopy.match import ObjTagged
    import feinsum as fnsm
    from functools import reduce
    from meshmode.feinsum_transformations import FEINSUM_TO_TRANSFORMS

    assert all(insn.tags_of_type(EinsumTag)
           for insn in t_unit.default_entrypoint.instructions
           if isinstance(insn, lp.MultiAssignmentBase)
           )

    einsum_tags = reduce(
        frozenset.union,
        (insn.tags_of_type(EinsumTag)
         for insn in t_unit.default_entrypoint.instructions),
        frozenset())
    for ensm_tag in sorted(einsum_tags,
       	    key=lambda x: sorted(x.orig_loop_nest)):
        if reduce(frozenset.union,
            (insn.reduction_inames()
                   for insn in (t_unit.default_entrypoint.instructions)
                   if ensm_tag in insn.tags),
             frozenset()):
            fused_einsum = fnsm.match_einsum(t_unit, ObjTagged(ensm_tag))
        else:
            # elementwise loop
            from meshmode.array_context import _get_elementwise_einsum
            fused_einsum = _get_elementwise_einsum(t_unit, ensm_tag)
        #print(fused_einsum)
        #print(fnsm.normalize_einsum(fused_einsum))
        #exit()
        try:
            fnsm_transform = FEINSUM_TO_TRANSFORMS[
                fnsm.normalize_einsum(fused_einsum)]
        except KeyError:
            try: 
                query_result = fnsm.query(fused_einsum,
                    queue.context,
                    err_if_no_results=True)
                print("Done querying")
                print(query_result)
                fnsm_transform = query_result[0].transform
                #1/0
            except RuntimeError:
                print("Could not find transformations for the following fused einsum")
                print(fused_einsum)
                raise RuntimeError

        t_unit = fnsm_transform(t_unit, insn_match=ObjTagged(ensm_tag))
        return t_unit

# Only works for subkernels that have no dependency on a prior subkernel
def autotune_standalone_subkernel(sk, queue, program_id=None, max_flop_rate=None, device_latency=None, device_memory_bandwidth=None, save_path=None):

    if save_path is None:
        save_path = "./hjson"

    einsum_types = list(get_einsum_types(sk))    
    if len(einsum_types) > 1:
        raise(ValueError("Cannot currently handle multiple einsum types in same subkernel"))
    est = einsum_types[0]

    use_ytopt = False

    handled_pairs = set([(2,1,),(3,2,),(2,2,),(2,3)])
    if (len(est[0]), len(est[1]),) in handled_pairs:
        if use_ytopt:
            input_space = createConfigSpace(queue, sk)
            ytopt_tuning(queue, sk, 0, input_space, program_id=program_id, max_flop_rate=max_flop_rate,
                             device_memory_bandwidth=device_memory_bandwidth,
                             device_latency=device_latency, timeout=30, save_path=save_path)
        else:
            print("ONLY TESTING THE FIRST 20 transformations")
            from random import shuffle
            trans_list_list = einsum3to2_kernel_tlist_generator_v2(queue, sk)
            shuffle(trans_list_list)

            tdict = parallel_autotune(sk, 0, trans_list_list[:10], program_id=program_id,
                        max_flop_rate=max_flop_rate, device_latency=device_latency,
                        device_memory_bandwidth=device_memory_bandwidth, save_path=save_path)
    else:
        print("Not tuning", sk.name)
        #raise(ValueError(f"Unhandled einsum type: {est}"))

    #return list(trans_list_list[0])

    # Should this return the transformations or just save them?

    return True

#def autotune_dependent_subkernel(subkernel, queue):


def autotune_parts(parts, queue):
    from generators import einsum3to2_kernel_tlist_generator_v2, einsum2to2_kernel_tlist_generator_v2
    #from parallel_autotuning_charm4py_v2 import parallel_autotune
    from parallel_autotuning_mpi4py_v2 import parallel_autotune
    from run_tests import run_single_param_set_v2
    from run_tests import generic_test

    cum_transformations = []
    transformations = None
    counter = 0
    for csk, sk in parts: # Subkernel and cumulative subkernel
        # Apply transformations to csk
        counter += 1
       
        # Determine einsum types of part and autotune based on those
        einsum_types = list(get_einsum_types(sk))
        if len(einsum_types) > 1:
            raise(ValueError("Cannot currently handle multiple einsum types in same subkernel"))

        #for instr in sk.default_entrypoint.instructions:
        #    if isinstance(instr, lp.Assignment):
        #        print(instr.dependency_names())
        #        #print(instr.assignee, type(instr.assignee))

        print("Einsum types", einsum_types)
        est = einsum_types[0]
        # Is a 3to2 einsum kernel
        if len(est[0]) == 2 and len(est[1]) == 1:
            trans_list_list = einsum3to2_kernel_tlist_generator_v2(queue, sk)
        elif len(est[0]) == 2 and len(est[1]) == 0:
            trans_list_list = einsum2to2_kernel_tlist_generator_v2(queue, sk)
        else:
            print(est)
            raise(ValueError("Unhandled einsum type"))

        #print(trans_list_list)
        # Test the kernel on the cumulative kernel
        #for entry in trans_list_list:
        #    for internal_entry in entry:
        #        print(internal_entry)
        #    print()
        #exit()
        #print(trans_list_list[0])
        # Just try the first transformation set for now
        cur_trans = cum_transformations + list(trans_list_list[0])

        #if counter == 2:
            #for entry in cur_trans:
            #    print(entry)
            #exit()

        #if counter != 2: # Avoid tuning the second part for the moment
        #autotune_result = run_single_param_set_v2(queue, csk, trans_list_list[1], generic_test)
        #    cum_transformations += list(autotune_result[1][:-1])
        
        #exit()
        #print(transformations)
        #exit()
        try:
            fs_csk = apply_feinsum_transformations(csk, queue)
            feinsum_tdict = run_single_param_set_v2(queue, fs_csk, [], generic_test)
            #tdict = parallel_autotune(csk, 0, trans_list_list)
            #print("Mine")
            #print(tdict)
            print("Feinsum")
            print(feinsum_tdict)

            exit()
        except RuntimeError:
            pass 
        #transformations = tdict["transformations"]
        # Chop off redundant add_inames_for_unused_hw_axes
        #cum_transformations += trans_list_list[0] # Just use the first one for now

    # Save transformations to file (should probably also save the metadata)

    #print(cum_transformations)
    #exit()

    #return cum_transformations
    return transformations

def get_subkernels(tunit, args):

    barriers = get_barriers(tunit)
    if len(barriers) > 1:
        phases = get_phases(tunit, barriers)
        cum_subkernels = generate_cumulative_subkernels(tunit, barriers, phases)
        subkernels = generate_subkernels(tunit, barriers, phases)
    else:
        subkernels = [tunit]
        cum_subkernels = [tunit]

    return list(zip(subkernels, cum_subkernels))

    """
    exit()

        #if any([isinstance(tag, EinsumTag) for tag in instr.tags]):
        #    print(str(instr))
        #barriers = []
        #if isinstance(instr, lp.BarrierInstruction):
        #    barriers.append(instr)
        #    print(str(instr))
            #print(instr.assignee_var_names(), instr.assignee_var_names(), instr.dependency_names(), instr.tags)
        #print(instr)

    for barrier in barriers:
        print(f"DEPENDS ON BARRIER {barrier}")
        for entry in phases[barrier]["instructions"]:
            print(str(entry))
        print()



    exit()



    print("ARGUMENTS")
    for entry in tunit.default_entrypoint.args:
        if isinstance(entry, lp.ArrayArg):
            print(entry.name, entry.shape, entry.tags)

    #for instr in tunit.default_entrypoint.instructions:


    print("TEMPORARIES")
    for name, val in tunit.default_entrypoint.temporary_variables.items():
       print(name, val.shape, val.address_space, val.tags)
       #print(entry.name, entry.shape)

    # If on read side and not on write side, then prefetch it.
    # Maybe have a tag to link otherwise independent inames so autotuner does not try to test them separately
    """


def has_internal_einsum_dependencies(tunit):
    einsum_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions if any([isinstance(tag, EinsumTag) for tag in instr.tags])}
    instr_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions}
    for esid, einsum in einsum_dict.items():
        es_deps = set(einsum.depends_on)
        while len(es_deps) > 0:
            dep_id = es_deps.pop()
            if dep_id in einsum_dict:
                return True
            else:
                es_deps |=  instr_dict[dep_id].depends_on
        
    return False 

def print_internal_einsum_dependencies(tunit):
    einsum_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions if any([isinstance(tag, EinsumTag) for tag in instr.tags])}
    instr_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions}
    for esid, einsum in einsum_dict.items():
        deps = []
        es_deps = set(einsum.depends_on)
        while len(es_deps) > 0:
            dep_id = es_deps.pop()
            if dep_id in einsum_dict:
                deps.append(dep_id)
                es_deps |=  instr_dict[dep_id].depends_on

        print(esid, "depends on the following einsums:", set(deps))


def get_pickled_tunits(directory):
    files = os.listdir(directory)
    tunit_dicts = []

    '''
    # Doesn't seem to capture the true call count
    call_count_dict = {}
    for filename in list(sorted(files)):
        if filename.startswith("call_count_") and (filename.endswith(".pickle") or filename.endswith(".pkl")):
            f = os.path.join(directory,filename)
            f = open(f, "rb")
            fdict = pickle.load(f)
            f.close()
            call_count_dict = fdict | call_count_dict # Give rank 0 the priority

    print(call_count_dict)
    exit()
    '''

    for num, filename in list(enumerate(sorted(files))):
        #print(num, filename)
        f = os.path.join(directory, filename)
        # Skip the massive kernel for now
        if os.path.isfile(f) and filename.startswith("prefeinsum") and (filename.endswith(".pickle") or filename.endswith(".pkl")):
            f = open(f, "rb")
            fdict = pickle.load(f)
            #pid = filename.split("_")[1]
            #print(fdict["tunit"])

            tunit_dicts.append((filename,fdict,))
            #tunit_dicts.append((filename,fdict,call_count_dict[pid]))

            #tunits.append((filename, tunit, args,))
            #tunit, args = pickle.load(f)
            f.close()

    #exit()
    return tunit_dicts


def get_lazy_einsum_info(tunit_dicts, hjson_dir=None):

    for filename, tunit_dict in tunit_dicts:
        tunit = tunit_dict["tunit"]
        print(tunit)
        args = tunit_dict["args"]
        sks = get_subkernels(tunit, args)
        #print(tunit.default_entrypoint)
        #contains_indirection(tunit)
        indirs = get_indirection_arrays(tunit)
        print(filename, len(sks), len(indirs) > 0)
        for sk, csk in sks:
            print(get_einsum_types(sk))
            print(sk.default_entrypoint.domains)
        #einsums = get_einsum_types(tunit)
        #for einsum in einsums:
        #    print("    ", einsum)
    #exit()

    # Count number of subkernels of each einsum type
    subkernel_counts = {}
    print("\nSubkernel information")
    pid_set = set()
    streaming_pid = set()
    einsum_3_to_2_pid = set()
    einsum_4_to_2_pid = set()
    einsum_5_to_3_pid = set()
    einsum_5_to_2_pid = set()
    other_einsum_pid = set()

    for filename, tunit_dict in tunit_dicts:

        # Restrict output to zero rank
        if "_0.pickle" in filename:
            tunit = tunit_dict["tunit"]
            args = tunit_dict["args"]
            sks = get_subkernels(tunit, args)
            #print("Number of subkernels", len(sks))

            for sk, csk in sks:

                #einsum_types = list(get_einsum_types(sk))
                einsum_counts = list(get_einsum_counts(sk).items())
                print("Einsum counts", einsum_counts)
                internal_deps = has_internal_einsum_dependencies(sk)
                #print_internal_einsum_dependencies(sk)

                indirection = len(get_indirection_arrays(sk)) > 0
                pid = unique_program_id(sk)
                pid_set |= {pid}

                #print(einsum_counts)
                if len(einsum_counts) > 1:
                    raise ValueError("There should not be multiple einsum types within a single subkernel")
                if len(einsum_counts) > 0:
                    einsum_type, count = einsum_counts[0]
                    non_red_axes = len(einsum_type[0])
                    red_axes = len(einsum_type[1])
                    total_axes = non_red_axes + red_axes
                    out_axes = total_axes - red_axes
                    key = (total_axes, out_axes, red_axes, count, indirection, internal_deps)

                    #if total_axes == 5 and non_red_axes == 2:
                    #    print(sk)
                    #    exit()
                    if red_axes == 0:
                        streaming_pid |= {pid}
                    elif total_axes == 3 and non_red_axes == 2:
                        einsum_3_to_2_pid |= {pid}
                    elif total_axes == 4 and non_red_axes == 2:
                        einsum_4_to_2_pid |= {pid}
                    elif total_axes == 5 and non_red_axes == 2:
                        einsum_5_to_2_pid |= {pid}
                    elif total_axes == 5 and non_red_axes == 3:
                        einsum_5_to_3_pid |= {pid}
                    else:
                        other_einsum_pid |= {pid}

                    """
                    data = None
                    if hjson_dir is not None:
                        fn = hjson_dir + f"/{pid}.hjson"
                        if exists(fn):
                            print(fn)
                            od = load_hjson(fn)
                            data = od["data"]["frac_roofline_flop_rate"]
                    print(pid, key, data)
                    """

                    if key in subkernel_counts:
                        subkernel_counts[key][0] += 1
                        subkernel_counts[key][1] |= set([sk.default_entrypoint.name])
                    else:
                        subkernel_counts[key] = [1, set([sk.default_entrypoint.name])]

    print("Rank zero info")

    print("\nSubkernel summary information")
    for key, val in subkernel_counts.items():
        print(key, val)

    print("Number of distinct subkernels", len(pid_set))
    print("Number of distinct streaming subkernels", len(streaming_pid))
    print("Number of distinct 3 to 2 einsums", len(einsum_3_to_2_pid))
    print("Number of distinct 4 to 2 einsums", len(einsum_4_to_2_pid))
    print("Number of distinct 5 to 2 einsums", len(einsum_5_to_2_pid))
    print("Number of distinct 5 to 3 einsums", len(einsum_5_to_3_pid))
    print("Number of distinct other einsums", len(other_einsum_pid))


def get_device_roofline_data(queue):
    import feinsum.empirical_roofline as er
    results_list = er.loopy_bandwidth_test(queue, fast=True, print_results=True, fill_on_device=True)
    device_latency = er.get_min_device_latency(results_list)
    loopy_bw = er.get_latency_adjusted_max_device_memory_bandwidth(results_list)
    clpeak_bw = er.get_max_bandwidth_clpeak(queue=queue)
    clpeak_flop_rate = er.get_max_flop_rate_clpeak(np.float64, queue=queue)    
    device_memory_bandwidth = max(loopy_bw, clpeak_bw)

    return device_latency, device_memory_bandwidth, clpeak_flop_rate


def autotune_standalone_subkernels(sk_list, save_path=None):

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    if save_path is None:
        save_path = "./hjson"

    if False:
        if not use_charm:
            if comm.Get_rank() == 0:
                # The latency is now obtained per-kernel so it probably needn't be obtained here.
                # Bandwidth microbenchmark could be improved to handle asymmetric numbers of reads and
                # writes.
                device_latency, device_memory_bandwidth, clpeak_flop_rate = get_device_roofline_data(queue)
            else:
                device_memory_bandwidth = None
                device_latency = None
                clpeak_flop_rate = None

            device_memory_bandwidth = comm.bcast(device_memory_bandwidth)
            device_latency = comm.bcast(device_latency)
            clpeak_flop_rate = comm.bcast(clpeak_flop_rate)
        else:
            device_latency, device_memory_bandwidth, clpeak_flop_rate = get_device_roofline_data(queue)
    else:
        device_memory_bandwidth = None
        device_latency = None
        clpeak_flop_rate = None

    for pid, sk, csk in sk_list:

        if False: # Feinsum autotuning
            feinsum_autotune(tunit, queue)
        else: # Eager-style autotuning

            os.makedirs(save_path, exist_ok=True)
            hjson_file = f"{save_path}/{pid}.hjson"
            if exists(hjson_file):
                print(f"A TUNE PROFILE ALREADY EXISTS: {hjson_file}")
            else:
                print(f"A TUNE PROFILE EXISTS NOT: {hjson_file}")

            #if True:
                einsum_counts = list(get_einsum_counts(sk).items())
                indirection = len(get_indirection_arrays(sk)) > 0
                if len(einsum_counts) > 0:

                    if len(einsum_counts) > 1:
                        raise ValueError("Subkernel has multiple einsum types")

                    einsum_type, einsum_count = einsum_counts[0]
                    non_red_axes = len(einsum_type[0])
                    red_axes = len(einsum_type[1])
                    total_axes = non_red_axes + red_axes
                    out_axes = total_axes - red_axes
                    
                    print("EINSUM INFO:", total_axes, non_red_axes, red_axes, indirection, einsum_count, pid)
                    
                    handled_pairs = set([(2,1,),(3,2,),(2,2,),(2,3)])
                    if (non_red_axes, red_axes,) in handled_pairs and einsum_count >= 0:
                        # Add indirection as a parameter?
                        autotune_standalone_subkernel(sk, queue, program_id=pid,
                                                      max_flop_rate=clpeak_flop_rate,
                                                      device_latency=device_latency,
                                                      device_memory_bandwidth=device_memory_bandwidth,
                                                      save_path=save_path)



def test_default_transforms(sk_list, save_path=None):

    if save_path is None:
        save_path = "default_transforms_hjson"

    os.makedirs(save_path, exist_ok=True)

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    from meshmode.array_context import FusionContractorArrayContext
    actx = FusionContractorArrayContext(queue)


    device_latency, device_memory_bandwidth, clpeak_flop_rate = get_device_roofline_data(queue)

    for pid, sk, csk in sk_list:
        print(f"Testing subkernel: {pid}")

        einsum_counts = list(get_einsum_counts(sk).items())
        indirection = len(get_indirection_arrays(sk)) > 0
        if len(einsum_counts) > 0:
            if len(einsum_counts) > 1:
                raise ValueError("Subkernel has multiple einsum types")

            einsum_type, einsum_count = einsum_counts[0]
            non_red_axes = len(einsum_type[0])
            red_axes = len(einsum_type[1])
            total_axes = non_red_axes + red_axes
            out_axes = total_axes - red_axes

            if not indirection and red_axes > 0:

                transformed_sk = actx.transform_loopy_program(sk)
                ret_dict = run_single_param_set_v2(queue, transformed_sk, [], generic_test,
                            max_flop_rate=clpeak_flop_rate, device_memory_bandwidth=device_memory_bandwidth,
                            device_latency=device_latency)
                   
                print(ret_dict["data"])
                # Should this functionality be a utility function
                hjson_file_str = save_path + f"/{pid}.hjson"
                out_file = open(hjson_file_str, "wt")
                hjson.dump(ret_dict, out_file, default=convert)
                out_file.close()

def test_feinsum_transforms(tunits):

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)


    for filename, tunit, args in tunits:
        sks = get_subkernels(tunit, args)
        for sk, csk, in sks:
            if len(get_indirection_arrays(tunit)) == 0 :
                try:
                    apply_feinsum_transformations(sk, queue)
                except RuntimeError:
                    print("Couldn't find transformation for", filename)


def compare_weighted_avg_frac_rooflines(directory, pid_dict):
    tuned_dir = directory + "/hjson"
    untuned_dir = directory + "/default_transforms_hjson"
    
    tuned_files = os.listdir(tuned_dir)
    untuned_files = os.listdir(untuned_dir)

    overlapping_files = set(tuned_files) &  set(untuned_files)

    tuned_data = []
    untuned_data = []

    print(len(overlapping_files))

    def get_data(files, d, pid_dict):

        data = []
        total_avg_exec_time = 0
        for filename in files:
            split_filename = filename.split(".")
            pid = split_filename[0]
            f = os.path.join(d, filename)
            #print(f)
            dct = load_hjson(f)
            data.append((pid, dct["data"],))
            total_avg_exec_time += pid_dict[pid]*(dct["data"]["avg_time"] - dct["data"]["device_memory_latency"])
            #total_avg_exec_time += dct["data"]["avg_time"] - dct["data"]["device_memory_latency"]

        weighted_avg_roofline = 0
        for pid, entry in data:
            weighted_avg_roofline += pid_dict[pid]*entry["frac_roofline_flop_rate"]*(entry["avg_time"] - entry["device_memory_latency"])/total_avg_exec_time
            #weighted_avg_roofline += entry["frac_roofline_flop_rate"]*(entry["avg_time"] - entry["device_memory_latency"])/total_avg_exec_time

        return weighted_avg_roofline

    print(overlapping_files)
    tuned_frac_roofline = get_data(overlapping_files, tuned_dir, pid_dict)
    untuned_frac_roofline = get_data(overlapping_files, untuned_dir, pid_dict)
    
    print(len(overlapping_files), len(untuned_files), untuned_frac_roofline, tuned_frac_roofline)



def collect_subkernels(tunit_dicts):

    out_list = []
    pid_counts = {}
    for filename, fdict in tunit_dicts:
        tunit = fdict["tunit"]
        args = fdict["args"]
        print(f"OBTAINING SUBKERNELS FROM: {filename}")
        sks = get_subkernels(tunit, args)

        for sk, csk in sks:
            # This may change the identifier so needs to be set beforehand
            assert sk.default_entrypoint.options.no_numpy
            assert sk.default_entrypoint.options.return_dict
            pid = unique_program_id(sk)
            out_list.append((pid, sk, csk,))

            # Could also do this with Collections.Counter
            if pid in pid_counts:
                pid_counts[pid] += 1
            else:
                pid_counts[pid] = 1

    return out_list, pid_counts



def main(arg):

    #dump_subkernels_from_pickled(None)
    #directory = "./pickled_programs_prediction"
    directories = [#"./pickled_programs_y3_prediction_order_1_eager",
                    "./pickled_programs_y3_prediction_order_2_lazy",
                    #"./pickled_programs_wave",
                    #"./pickled_programs_prediction_order_1",
                    #"./pickled_programs_y3_prediction_order_1",
                    #"./pickled_programs_y3_prediction_order_3",
                    #"./pickled_programs_prediction_order_2",
                    #"./pickled_programs_prediction_order_3",
                    #"./pickled_programs_prediction_order_4"
                  ]
    
    # Could sort subkernels by dimensions, then use the maximum long axis
    # for the kernels that share the other dimensions as the dimension
    # length for tuning purposes. Then map the subkernels keys
    # to that hjson file (or just duplicate the hjson file for each subkernel
    # key). Need to count that the number of inames is the same though. And need to
    # figure out the element iname

    # Or just have the size be a parameter in the Bayesian optimization space.

    for directory in directories:
        save_path = directory + "/hjson"
        tunit_dicts = get_pickled_tunits(directory)
        print("Done collecting tunits")
        # ID changes based on whether python was run with -O
        sk_list, pid_dict = collect_subkernels(tunit_dicts)
        """
        for item in sk_list:
            sk = item[1].default_entrypoint
            a_exprs = [insn.expression for insn in sk.instructions if isinstance(insn, lp.Assignment)]# and isinstance(insn.expression, lp.symbolic.Reduction)]
            if len(a_exprs) != len(set(a_exprs)):

                print(sk.name, len(a_exprs), len(set(a_exprs)))
                if False:#sk.name == "unfiltered_rhs_15":
                    for entry in a_exprs:
                        print(entry)
                    #print(sk)

            #if item[1].default_entrypoint.name == "unfiltered_rhs_5":
            #    print(item[1].default_entrypoint)
            #    exit()
        exit()
        """
        print("Done collecting subkernels")
        get_lazy_einsum_info(tunit_dicts, hjson_dir=save_path)
        #exit()

        #test_default_transforms(sk_list, save_path=directory + "/default_transforms_hjson")

        autotune_standalone_subkernels(sk_list, save_path=save_path)

        #compare_weighted_avg_frac_rooflines(directory, pid_dict)

    exit() 

if __name__ == "__main__":
    if use_charm:
        charm.start(main)
        exit() # charm.exit freezes the program
        charm.exit()
    else:
        main(0)
