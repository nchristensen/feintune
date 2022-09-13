import pickle
import loopy as lp
from pytools.tag import Tag
from meshmode.array_context import EinsumTag
import pyopencl as cl
import os

from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
from charm4py.pool import PoolScheduler, Pool
from charm4py.charm import Charm, CharmRemote

import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD


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

        knl = lp.make_kernel(domains, instructions, kernel_data=new_args)
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
        knl = lp.make_kernel(domains, instructions, kernel_data=new_args)
        subkernels.append(knl)
    return subkernels


# Obtain non-reduction and reduction inames 
def get_einsum_types(knl):
    einsums = []
    for instr in knl.default_entrypoint.instructions:
        if isinstance(instr, lp.Assignment):
            for tag in instr.tags:
                if isinstance(tag, EinsumTag):
                    if isinstance(instr.expression, lp.symbolic.Reduction):
                        einsums.append((instr.within_inames, instr.expression.inames,))
                    else:
                        einsums.append((instr.within_inames, (),))
                    
    
    return frozenset(einsums)


def dump_subkernels_from_pickled(arg):

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    directory="./pickled_programs"
    files = os.listdir(directory)
    for num, filename in enumerate(files):
        f = os.path.join(directory, filename)
        # Skip the massive kernel for now
        if os.path.isfile(f) and (filename.endswith(".pickle") or filename.endswith(".pkl")) and num != 3:
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


def autotune_parts(parts, queue):
    from generators import einsum3to2_kernel_tlist_generator_v2, einsum2to2_kernel_tlist_generator_v2
    #from parallel_autotuning_charm4py_v2 import parallel_autotune
    from parallel_autotuning_mpi4py_v2 import parallel_autotune
    from run_tests import run_single_param_set_v2
    from run_tests import generic_test

    cum_transformations = []
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
        #    autotune_result = run_single_param_set_v2(queue, csk, cur_trans, generic_test)
        #    cum_transformations += list(autotune_result[1][:-1])

        #print(transformations)
        #exit()
        transformations = parallel_autotune(csk, 0, trans_list_list)
        # Chop off redundant add_inames_for_unused_hw_axes
        #cum_transformations += trans_list_list[0] # Just use the first one for now

    # Save transformations to file (should probably also save the metadata)

    print(cum_transformations)
    #exit()

    return cum_transformations


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

if __name__ == "__main__":
    dump_subkernels_from_pickled(None)
    #charm.start(dump_subkernels_from_pickled)
    #print(result)
    #charm.exit()


