import pickle
import loopy as lp
from pytools.tag import Tag
from meshmode.array_context import EinsumTag
import pyopencl as cl
import os
from os.path import exists
from utils import unique_program_id

#import mpi4py.MPI as MPI
#comm = MPI.COMM_WORLD

use_charm=True
if use_charm:
    from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
    from charm4py.pool import PoolScheduler, Pool
    from charm4py.charm import Charm, CharmRemote
    from parallel_autotuning_charm4py_v2 import parallel_autotune
else:
    from parallel_autotuning_mpi4py_v2 import parallel_autotune



from generators import einsum3to2_kernel_tlist_generator_v2
from run_tests import run_single_param_set_v2
from run_tests import generic_test

class IsDOFArray(Tag):
    pass

# Map a domain to a tuple for its inames
def get_domain_list(tunit):
    domains = tunit.default_entrypoint.domains
    import islpy
    domain_list = []
    for domain in domains:
        #print(domain.get_var_names(islpy.dim_type.all))
        domain_names = frozenset([key.name for key in domain.get_id_dict().keys()])
        domain_list.append((domain_names, domain,))
    return domain_list

# Get the barriers to divide computation into phases
def get_barriers(tunit):
    barriers = [None]
    for instr in tunit.default_entrypoint.instructions:

        #if any([isinstance(tag, EinsumTag) for tag in instr.tags]):
        #    print(str(instr))
        if isinstance(instr, lp.BarrierInstruction) and instr.synchronization_kind == "global":
            barriers.append(instr.id)
    return barriers


# Get the barriers to divide computation into phases
def get_phases(tunit, barriers, domain_list):

    # Should a phase be an object
    phase_lists = [{"domains": frozenset(), "instructions": [], "args": frozenset()} for i in range(len(barriers) + 1)]
    phases = dict(zip(barriers, phase_lists))
    #print(phases)

    for instr in tunit.default_entrypoint.instructions:

        #print(instr.within_inames)
        #print(domain_dict[instr.within_inames])
        #if not (isinstance(instr, lp.BarrierInstruction) and instr.synchronization_kind == "global"):
        dbarrier = None
        for entry in instr.depends_on:
            if entry in barriers:
                dbarrier = entry
                break

        phases[dbarrier]["instructions"].append(instr)
        phases[dbarrier]["domains"] = instr.within_inames | phases[dbarrier]["domains"]


    # Replace the text domain names with the actual domain objects
    for dbarrier in barriers:
        
        domain_ids = phases[dbarrier]["domains"]
        phases[dbarrier]["domains"] = []

        for domain_names_set, domain in domain_list:
            if domain_ids <= domain_names_set:
                phases[dbarrier]["domains"].append(domain)


        #print(phases[dbarrier]["domains"])
    return phases

# Strip off the dependencies on global barriers and other phases
def strip_unused_dependencies(instructions):
    phase_instruction_ids = [instruction.id for instruction in instructions]

    new_instructions = []
    #print(phase_instruction_ids)
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

        #print(new_dependencies, instruction.depends_on)
        new_instruction = instruction.copy()
        new_instruction.depends_on = frozenset(new_dependencies)
        new_instructions.append(new_instruction)

    # Strip off unused barrier instructions
    new_new_instructions = []
    for instruction in new_instructions:
        if not (isinstance(instruction, lp.BarrierInstruction) and barrier_dep_count[instruction.id] == 0):
            new_new_instructions.append(instruction)

    return new_new_instructions


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

        name = tunit.default_entrypoint.name + f"{cur_phase}_cum"
        knl = lp.make_kernel(domains, instructions, kernel_data=new_args, name=name)
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
        name = tunit.default_entrypoint.name + f"{cur_phase}"
        knl = lp.make_kernel(domains, instructions, kernel_data=new_args, name=name)
        subkernels.append(knl)
    return subkernels


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
                    
    
    return einsums

def get_einsum_counts(knl):
    from collections import Counter
    counter = Counter(get_einsums(knl))
    print(counter.items())
    return counter

# Obtain non-reduction and reduction inames 
def get_einsum_types(knl):
    return frozenset(get_einsums(knl))

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
            #tunit, args = pickle.load(f)
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

def feinsum_autotune(tunit, queue):
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
def autotune_standalone_subkernel(sk, queue, max_flop_rate=None, device_latency=None, device_memory_bandwidth=None):
    einsum_types = list(get_einsum_types(sk))    

    if len(einsum_types) > 1:
        raise(ValueError("Cannot currently handle multiple einsum types in same subkernel"))

    est = einsum_types[0]

    if len(est[0]) == 2 and len(est[1]) == 1:
        trans_list_list = einsum3to2_kernel_tlist_generator_v2(queue, sk)
    #elif len(est[0]) == 2 and len(est[1]) == 0:
    #    trans_list_list = einsum2to2_kernel_tlist_generator_v2(queue, sk)
    else:
        print(est)
        raise(ValueError("Unhandled einsum type"))

    tdict = parallel_autotune(sk, 0, trans_list_list, max_flop_rate=max_flop_rate, device_latency=device_latency,
            device_memory_bandwidth=device_memory_bandwidth)
    
    transformations = tdict["transformations"]
    return transformations

    #return list(trans_list_list[0])


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

        for instr in sk.default_entrypoint.instructions:
            if isinstance(instr, lp.Assignment):
                print(instr.dependency_names())
                #print(instr.assignee, type(instr.assignee))

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

    #file_path = "./pickled_programs/03dccf17ebb345c3.pickle"
    #file_path = "./pickled_programs/03dcff6d7c9ed451.pickle"
    #f = open(file_path, "rb")
    #tunit, args = pickle.load(f)
    #f.close()

    #print(tunit)
    #print(args)

    ### Apply tags

    # Just slap the tag on the arrays for now. Will eventually need to figure out how to propagate the tags
    """
    new_args = []
    for entry in tunit.default_entrypoint.args:
        if entry.shape == (1348732, 15):
            entry = entry.tagged(IsDOFArray())
        new_args.append(entry)

    # Cheat for now, will eventually need to figure out how to propagate this from array arguments.
    # Maybe this can be done in the DAG before generation though
    new_temps = {}
    for name, val in tunit.default_entrypoint.temporary_variables.items():
        if val.shape == (1348732, 15):
            val = val.tagged(IsDOFArray())
        print(val)
        new_temps[name] = val

    tunit = tunit.with_kernel(tunit.default_entrypoint.copy(args=new_args, temporary_variables=new_temps))
    """

    ### End tag application

    domain_list = get_domain_list(tunit)
    barriers = get_barriers(tunit)
    phases = get_phases(tunit, barriers, domain_list)
    cum_subkernels = generate_cumulative_subkernels(tunit, barriers, phases)
    subkernels = generate_subkernels(tunit, barriers, phases)

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

## Kaushik's indirection finder code
import loopy as lp
from loopy.symbolic import CombineMapper, DependencyMapper
from typing import FrozenSet
import pymbolic.primitives as prim
import numpy as np

"""
class IndirectionFinder(CombineMapper):
    def __init__(self, all_inames: FrozenSet[str]):
        super().__init__()
        self.all_inames = all_inames

    def combine(self, values):
        return any(values)

    def map_subscript(self, expr):
        return not all(((isinstance(idx, prim.Variable)
                         and idx.name in self.all_inames)
                        or np.isscalar(idx))
                       for idx in expr.index_tuple)

    def map_variable(self, expr):
        return False
"""

class MyDepMapper(DependencyMapper):
    def map_subscript(self, expr, should_record=False):

        super_subscript = super().map_subscript(expr, should_record=True)
        aggregate = self.rec(expr.aggregate, should_record=True)

        #print(expr, expr.aggregate)
        #print(super_subscript, aggregate)

        #print(super_subscript, super_subscript - aggregate) #not should_record else frozenset()
        if not should_record:
            retval = super_subscript - aggregate #not should_record else frozenset()
        else:
            retval = super_subscript
        #print(retval)
        return retval

    def map_variable(self, expr, should_record=False):
        #print("MAP VARIABLE", should_record)
        return super().map_variable(expr, should_record=should_record) if should_record else frozenset()

    #def map_constant(self, expr, should_record=False):
        #print("MAP CONSTANT", expr)
        #return frozenset()


tunit = lp.make_kernel(
    "{[i, j]: 0<=i,j<10}",
    """
    y[map[i], j] = j*sin(x[i, map[2*i]]) {id=foo}
    """,
    [lp.GlobalArg("x,y", shape=None),
     ...],
    lang_version=(2018, 2))

knl = tunit.default_entrypoint
dep_mapper = MyDepMapper(include_subscripts=False)
result = set()
for insn in knl.instructions:
    result.update(dep_mapper(insn.expression, should_record=False))
print("RHS index deps are:", result)


def get_index_deps(tunit):
    knl = tunit.default_entrypoint
    dep_mapper = MyDepMapper(include_subscripts=False)
    result = set()
    for insn in knl.instructions:
        if not isinstance(insn, lp.BarrierInstruction):
            result.update(dep_mapper(insn.expression, should_record=False))
    #print("RHS index deps are:", result)
    retval =  frozenset([var.name for var in result]) 
    return retval

def get_indirection_arrays(tunit):
    index_deps = get_index_deps(tunit)
    inames = frozenset(tunit.default_entrypoint.inames.keys())
    indirection_arrays = index_deps - (index_deps & inames)
    #print("Indirection arrays:", indirection_arrays)
    return indirection_arrays  

# Doesn't work
"""
def contains_indirection(tunit):
    knl = tunit.default_entrypoint
    indirection_finder = IndirectionFinder(knl.all_inames())
    if any(indirection_finder(insn.expression)
           for insn in knl.instructions):
        print("Kernel contains indirection")
    else:
        print("Kernel does *NOT* contain indirection")
"""

def get_pickled_tunits(directory):
    files = os.listdir(directory)
    tunits = []
    for num, filename in list(enumerate(sorted(files))):
        #print(num, filename)
        f = os.path.join(directory, filename)
        # Skip the massive kernel for now
        if os.path.isfile(f) and filename.startswith("prefeinsum") and (filename.endswith(".pickle") or filename.endswith(".pkl")):
            f = open(f, "rb")
            tunit, args = pickle.load(f)

            tunits.append((filename, tunit, args,))
            #tunit, args = pickle.load(f)
            f.close()


    return tunits

def get_lazy_einsum_info(tunits):
    for filename, tunit, args in tunits:
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

    # Count number of subkernels of each einsum type
    subkernel_counts = {}
    for filename, tunit, args in tunits:
        sks = get_subkernels(tunit, args)
        for sk, csk in sks:
            #einsum_types = list(get_einsum_types(sk))
            einsum_counts = list(get_einsum_counts(sk).items())
            indirection = len(get_indirection_arrays(sk)) > 0
            #print(einsum_counts)
            if len(einsum_counts) > 1:
                raise ValueError("There should not be multiple einsum types within a single subkernel")
            if len(einsum_counts) > 0:
                einsum_type, count = einsum_counts[0]
                non_red_axes = len(einsum_type[0])
                red_axes = len(einsum_type[1])
                total_axes = non_red_axes + red_axes
                out_axes = total_axes - red_axes
                key = (total_axes, out_axes, red_axes, count, indirection)
                if key in subkernel_counts:
                    subkernel_counts[key] += 1
                else:
                    subkernel_counts[key] = 1


    for key, val in subkernel_counts.items():
        print(key, val)

def autotune_standalone_subkernels(tunits):
    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Doesn't really matter if all of the processes do this,
    # the rank 0 numbers will be used
    # Would possibly be more accurate to use the minimum latency ever seen
    # and the maximum bandwidth ever seen
    if not use_charm:
        if comm.Get_rank() == 0:
            import feinsum.empirical_roofline as er
            results_list = er.loopy_bandwidth_test(queue, fast=True, print_results=True, fill_on_device=True)
            device_latency = er.get_min_device_memory_latency(results_list)
            loopy_bw = er.get_latency_adjusted_max_device_memory_bandwidth(results_list)
            clpeak_bw = er.get_max_bandwidth_clpeak(queue=queue)
            clpeak_flop_rate = er.get_max_flop_rate_clpeak(np.float64, queue=queue)    
            device_memory_bandwidth = max(loopy_bw, clpeak_bw)
        else:
            device_memory_bandwidth = None
            device_latency = None
            clpeak_flop_rate = None

        device_memory_latency = comm.bcast(device_memory_bandwidth)
        device_latency = comm.bcast(device_latency)
        clpeak_flop_rate = comm.bcast(clpeak_flop_rate)

        #device_latency, inverse_bandwidth = get_alpha_beta_model(results_list)
        #device_memory_bandwidth = 1/inverse_bandwidth
    else:
        import feinsum.empirical_roofline as er
        results_list = er.loopy_bandwidth_test(queue, fast=True, print_results=True, fill_on_device=True)
        device_latency = er.get_min_device_memory_latency(results_list)
        loopy_bw = er.get_latency_adjusted_max_device_memory_bandwidth(results_list)
        clpeak_bw = er.get_max_bandwidth_clpeak(queue=queue)
        clpeak_flop_rate = er.get_max_flop_rate_clpeak(np.float64, queue=queue)    
        device_memory_bandwidth = max(loopy_bw, clpeak_bw)

    for filename, tunit, args in tunits:
        print(f"TESTING TUNIT: {filename}")
        sks = get_subkernels(tunit, args)
        for sk, csk in sks:
            pid = unique_program_id(sk)
            os.makedirs(os.getcwd() + "/hjson", exist_ok=True)
            hjson_file = f"./hjson/{pid}.hjson"
            if exists(hjson_file):
                print("A TUNE PROFILE ALREADY EXISTS: {filename}")
            else:
                print(f"A TUNE PROFILE EXISTS NOT: {filename}")
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
                    
                    print("EINSUM INFO:", total_axes, non_red_axes, red_axes, indirection, einsum_count)
                    if not indirection and out_axes == 2 and total_axes == 3 and einsum_count <= 100:
                        print(sk)
                        autotune_standalone_subkernel(sk, queue, max_flop_rate=clpeak_flop_rate,
                                device_latency=device_latency, device_memory_bandwidth=device_memory_bandwidth)

    exit()
    #test_feinsum_transforms(tunits)

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

def main(arg):
    #dump_subkernels_from_pickled(None)
    #directory = "./pickled_programs_prediction"
    directory = "./pickled_programs_prediction_order_1"
    tunits = get_pickled_tunits(directory)
    #print(len(tunits))
    #get_lazy_einsum_info(tunits)
    autotune_standalone_subkernels(tunits)
   

if __name__ == "__main__":
    if use_charm:
        charm.start(main)
        exit() # charm.exit freezes the program
        charm.exit()
    else:
        main(0)
