from .apply_transformations import get_einsums, get_einsum_counts, get_einsum_types
from feintune.generators import einsum3to2_kernel_tlist_generator_v2
from feintune.run_tests import run_single_param_set_v2, generic_test, get_knl_flops
import numpy as np
import pickle
import loopy as lp
from pytools.tag import Tag
from meshmode.array_context import EinsumTag
import pyopencl as cl
import os
from os.path import exists
from feintune.utils import unique_program_id, convert, load_hjson, dump_hjson, get_domain_list, get_indirection_arrays
import hjson
from feintune.generators import createConfigSpace
from time import time
import logging

# Ignore some annoying warnings relating to pandas.concat
import warnings
warnings.filterwarnings("ignore")

comm = None

use_charm = False
if use_charm:
    from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
    from charm4py.pool import PoolScheduler, Pool
    from charm4py.charm import Charm, CharmRemote
    from feintune.parallel_autotuning_charm4py_v2 import parallel_autotune
else:
    import mpi4py
    mpi4py.rc.initialize = False
    import mpi4py.MPI as MPI
    # Check if run with an mpi runner, and initialize MPI if so.
    # Currently need to set this to True to use mpi
    if not MPI.Is_initialized():
        MPI.Init()
        comm = MPI.COMM_WORLD
    from feintune.parallel_autotuning_mpi4py_v2 import parallel_autotune

from feintune.ytopt_autotuning import ytopt_tuning

logger = logging.getLogger(__name__)

# Get the barriers to divide computation into phases
def get_barriers(tunit):
    barriers = [None]
    for instr in tunit.default_entrypoint.instructions:
        if isinstance(instr, lp.BarrierInstruction) and instr.synchronization_kind == "global":
            barriers.append(instr.id)
    # print("Number of global barriers", len(barriers))
    return barriers


# Get the barriers to divide computation into phases
def get_phases(tunit, barriers):

    # Should a phase be an object?
    phase_lists = [{"domains": frozenset(), "within_inames": frozenset(
    ), "instructions": [], "args": frozenset()} for i in range(len(barriers) + 1)]
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

        # print(len(phases[dbarrier]["domains"]))

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
    barriers = [instr.copy() for instr in macrokernel.default_entrypoint.instructions if isinstance(
        instr, lp.BarrierInstruction) and instr.synchronization_kind == "global"]

    # print("SUBKERNEL INAMES")
    # for sk in subkernels:
    #    print(sorted(lp.preprocess_kernel(sk).default_entrypoint.inames.keys()))

    iname_to_tag = []
    for sk in subkernels:
        for iname, iname_obj in sk.default_entrypoint.inames.items():
            for tag in iname_obj.tags:
                iname_to_tag.append((iname, tag,))

    # The old scheduler apparently needs explicit nosyncs.
    # Will this cause problems?
    # Ideally, the new scheduler would handle ilp or
    # could be used on a per-schedule basis.
    # """
    for isk, sk in enumerate(subkernels):
        es_deps_dict = get_internal_einsum_dependencies(sk)
        for sink, sources in es_deps_dict.items():
            for source in sources:
                sk = lp.add_nosync(sk, "global", source,
                                   sink, bidirectional=False)
        subkernels[isk] = sk  # lp.linearize(lp.preprocess(sk))
        # num = isk - 1
    # """

    assert len(subkernels) - 1 == len(barriers)
    new_instructions_all = []
    new_instructions = subkernels[0].default_entrypoint.instructions
    new_instructions_all += new_instructions
    domains = subkernels[0].default_entrypoint.domains
    temporaries = macrokernel.default_entrypoint.temporary_variables
    temporaries |= subkernels[0].default_entrypoint.temporary_variables

    for barrier, subkernel in zip(barriers, subkernels[1:]):

        barrier.depends_on = frozenset(
            [instr.id for instr in new_instructions])
        new_instructions_all.append(barrier)
        domains = domains + subkernel.default_entrypoint.domains
        temporaries |= subkernel.default_entrypoint.temporary_variables

        new_instructions = []
        for sk_instruction in subkernel.default_entrypoint.instructions:
            sk_instruction.depends_on |= frozenset([barrier.id])
            new_instructions.append(sk_instruction)
        new_instructions_all += new_instructions

    # Also need to copy the domains and inames
    new_macrokernel = macrokernel.default_entrypoint.copy(instructions=new_instructions_all,
                                                          domains=domains,
                                                          temporary_variables=temporaries)
    # new_macrokernel = lp.preprocess_program(new_macrokernel)
    # new_macrokernel = lp.linearize(new_macrokernel)

    new_macrokernel = lp.tag_inames(
        new_macrokernel, iname_to_tag, ignore_nonexistent=True)
    return macrokernel.with_kernel(new_macrokernel)

    # new_macrokernel = lp.make_kernel(domains,
    #                                 new_instructions_all,
    #                                 kernel_data=list(macrokernel.default_entrypoint.args) + list(temporaries.values()),
    #                                 name=macrokernel.default_entrypoint.name)

    # options = macrokernel.default_entrypoint.options
    # options = lp.Options(no_numpy=True, return_dict=True,
    #                     enforce_variable_access_ordered=True, enforce_array_accesses_within_bounds=True, insert_gbarriers=True)

    # new_macrokernel = lp.set_options(new_macrokernel, options)

    # return new_macrokernel
    # new_macrokernel = new_macrokernel.default_entrypoint
    # print(new_macrokernel.default_entrypoint.name)
    # exit()

# FIXME: Needs to look at the csv file instead of the hjson file
# FIXME: Just return a list of transform dictionaries. --Why?

# Searches for a transform file matching each macrokernel's subkernel's program id
# in save_path. Uses those transformations if a transform file is found
# otherwise uses the transformations of PrefusedFusionContractorArrayContext.
# Can optionally perform tuning.

#i#ef transform_macrokernel_actx(tunit_dict, actx):
#    tunit_dict[1]["tunit"]

def transform_macrokernel(tunit_dict, save_path, in_actx=None, tune=False, device_latency=None, device_memory_bandwidth=None, peak_flop_rate=None):

    logger.info("Transforming macrokernel")
    # macrokernels_to_tune = ["rhs"]
    sk_list, pid_counts = collect_subkernels([tunit_dict])
    # macrokernels_to_tune = ["frozen_result"]
    # if in_actx is None:# and tunit_dict[1]["tunit"].default_entrypoint.name in macrokernels_to_tune:
    if tune:
        autotune_standalone_subkernels(sk_list, save_path=save_path, device_latency=device_latency,
                                       device_memory_bandwidth=device_memory_bandwidth, peak_flop_rate=peak_flop_rate)
        logger.info("Done tuning macrokernel")

    sk_to_avoid = [  # "frozen_inv_metric_deriv_vol_0",
        # "frozen_inv_metric_deriv_vol_1",
        # "frozen_inv_metric_deriv_vol_2",
        # "frozen_inv_metric_deriv_vol_3"
    ]
    tunit_to_avoid = []  # ["frozen_inv_metric_deriv_vol"]#["rhs","frozen_inv_metric_deriv_vol"]

    # if tunit_dict[1]["tunit"].default_entrypoint.name in tunit_to_avoid:
    #    print(len(sk_list))
    #    for sk in sk_list:
    #        print(get_einsum_types(sk[1]))
    #    exit()

    # Should probably use in_actx instead.
    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    from meshmode.array_context import PrefusedFusionContractorArrayContext
    actx = PrefusedFusionContractorArrayContext(queue)

    transformed_subkernels = []
    for sk_dict in sk_list:
        pid = sk_dict["pid"]
        sk = sk_dict["sk"]

        # TODO If the transformation selected is one that timed out, should
        # use the default transformations instead.

        if in_actx is not None:
            default_transformed_sk = in_actx.transform_loopy_program(sk)
            transformed_subkernels.append((pid, default_transformed_sk,))
        else:

            # pid = unique_program_id(sk)
            # Tune the subkernel
            hjson_file_str = save_path + "/" + pid + ".hjson"
            if exists(hjson_file_str) and  \
               tunit_dict[1]["tunit"].default_entrypoint.name not in tunit_to_avoid:
               # sk.default_entrypoint.name not in sk_to_avoid and \
                print("Found", hjson_file_str)
                hjson = load_hjson(hjson_file_str)
                print("HJSON", hjson_file_str, hjson)
                from .apply_transformations import apply_transformation_list
                tsk = apply_transformation_list(
                    sk, hjson["transformations"])[0]
                transformed_subkernels.append((pid, tsk,))

            else:
                print("Can't find", hjson_file_str)
                # Should probably apply the default transformations
                # Currently fails

                logger.info("ACTX TRANSFORMING")
                tsk = actx.transform_loopy_program(sk)
                transformed_subkernels.append((pid, tsk,))
                logger.info("ACTX DONE TRANSFORMING")
                # transformed_subkernels.append(sk)
            # Transform ...

            # transformed_subkernels.append(transformed_subkernel)

        # exit()

        # print("PRE-TRANSFORMATION")
    # print(tunit_dict[1]["tunit"])
    # print("REASSEMBLING")

    # orig_tunit = tunit_dict[1]["tunit"]
    # orig_tunit = actx.transform_loopy_program(tunit_dict[1]["tunit"])
    # orig_tunit = lp.preprocess_program(actx.transform_loopy_program(orig_tunit))
    # print(orig_tunit)
    # if orig_tunit.default_entrypoint.name == "rhs":
    #    breakpoint()

    # print("ORIGINAL TUNIT")
    # orig_tunit = lp.linearize(orig_tunit)
    # if orig_tunit.default_entrypoint.name == "rhs":
    #    print(orig_tunit)
    #    print(orig_tunit.default_entrypoint.schedule)
    #    exit()

    # transformed_subkernels_2 = []
    # for pid, tsk in transformed_subkernels:
    #    assert lp.has_schedulable_iname_nesting(tsk)
    #    tsk = lp.preprocess_program(tsk)
        # The old scheduler seems to have a problem with unbarriered internal dependencies. Need to add no_sync_with
        # or find a way to use the schedule from the component subkernels.
    #    tsk = lp.linearize(tsk)
    #    transformed_subkernels_2.append((pid,tsk,))
    # transformed_subkernels = transformed_subkernels_2

    transformed_tunit = assemble_transformed_macrokernel(
        tunit_dict[1]["tunit"], [tsk[1] for tsk in transformed_subkernels])
    # assert lp.has_schedulable_iname_nesting(tunit_dict[1]["tunit"])
    assert lp.has_schedulable_iname_nesting(transformed_tunit)

    # transformed_tunit = lp.preprocess_program(transformed_tunit)
    print("NEW TUNIT")
    # print(transformed_tunit)

    # transformed_tunit = lp.linearize(transformed_tunit)
    # transformed_tunit = lp.save_and_reload_temporaries(transformed_tunit)
    print("DONE REASSEMBLING")

    print("POST_TRANSFORMATION")
    # print(transformed_tunit)
    # print(transformed_tunit.default_entrypoint.schedule)
    # if transformed_tunit.default_entrypoint.name == "rhs":
    #    exit()
    print("END OF KERNEL")

    # if transformed_tunit.default_entrypoint.name == "rhs":
    #    run_single_param_set_v2(actx.queue, transformed_tunit, [], generic_test)
    #    exit()
    return transformed_tunit, transformed_subkernels




# Create a subkernel with the domains and instructions of each cumulative phase
def generate_cumulative_subkernels(tunit, barriers, phases):
    subkernels = []
    for cur_phase in range(len(barriers)):
        # print(f"BARRIER {barriers[cur_phase]}")
        domains = []
        instructions = []
        for i in range(cur_phase + 1):
            domains += phases[barriers[i]]["domains"]
            # print(domains)
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
                new_args.append(
                    tunit.default_entrypoint.temporary_variables[entry])

        name = tunit.default_entrypoint.name + f"_{cur_phase}_cum"
        knl = lp.make_kernel(domains, instructions,
                             kernel_data=new_args, name=name)
        knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
        subkernels.append(knl)
    return subkernels

# Create a subkernel with the domains and instructions of each single phase


def generate_subkernels(tunit, barriers, phases):
    subkernels = []
    for cur_phase in range(len(barriers)):
        # print(f"BARRIER {barriers[cur_phase]}")
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
                temp_args.append(
                    tunit.default_entrypoint.temporary_variables[entry])

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
                                   alignment=temp.alignment, tags=temp.tags)  # Any others needed?
                new_args.append(arg)
            else:
                new_temp_args.append(temp)

        new_args += new_temp_args
        name = tunit.default_entrypoint.name + f"_{cur_phase}"
        # print("DOMAINS")
        # print(domains)
        # for domain in domains:
        #    print(domain)
        # for instruction in instructions:
        #    print(instruction)
        knl = lp.make_kernel(domains, instructions,
                             kernel_data=new_args, name=name)
        knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
        # if knl.default_entrypoint.name == "unfiltered_rhs_5" and ("pt_temp_119" in knl.default_entrypoint.args or "pt_temp_119" in knl.default_entrypoint.temporary_variables.values()):
        #    print(knl)
        #    exit()
        subkernels.append(knl)
    return subkernels


def dump_subkernels_from_pickled(arg):

    platforms = cl.get_platforms()
    cl_ctx = cl.Context(
        dev_type=cl.device_type.GPU,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(cl_ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    directory = "./pickled_programs/pickled_programs_eighthX_order_4"
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
    # exit()


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
        # print(fused_einsum)
        # print(fnsm.normalize_einsum(fused_einsum))
        # exit()
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
                # 1/0
            except RuntimeError:
                print("Could not find transformations for the following fused einsum")
                print(fused_einsum)
                raise RuntimeError

        t_unit = fnsm_transform(t_unit, insn_match=ObjTagged(ensm_tag))
        return t_unit

# Only works for subkernels that have no dependency on a prior subkernel


def autotune_standalone_subkernel(sk, queue, program_id=None, normalized_program_id=None, max_flop_rate=None, device_latency=None, device_memory_bandwidth=None, save_path=None):

    print("AUTOTUNING STANDALONE SUBKERNEL")

    if save_path is None:
        save_path = "./hjson"

    einsum_types = list(get_einsum_types(sk))
    if len(einsum_types) > 1:
        raise (ValueError(
            "Cannot currently handle multiple einsum types in same subkernel"))
    est = einsum_types[0]

    use_ytopt = True
    indirection = len(get_indirection_arrays(sk)) > 0

    handled_pairs = set([(2, 1,), (3, 2,), (2, 2,), (2, 3)])
    timeout = 20*60
    platform_id = queue.device.platform.name  # Set to 1 to use Nvidia OpenCL on Monza. Need more robust way.

    if (len(est[0]), len(est[1]),) in handled_pairs and not indirection:
        if use_ytopt:
            # Won't work with charm. But the charm4py executor is broken anyway.
            eval_str = "local_libensemble"
            #eval_str = "mpi_libensemble_subprocess" # Problematic for opencl on summit
            #eval_str = "mpi_libensemble" # This is the only one that works on summit
            """
            if comm.Get_size() <= 1:
                #eval_str = "local_libensemble"
                eval_str = "threadpool"
                #eval_str = "processpool" # Seems to be busted. "Exception: cannot pickle 'pyopencl._cl._ErrorRecord' object"
                #eval_str = "subprocess" # Also errors out.
            elif comm.Get_size() >= 3: # Breaks on Lassen.
                #eval_str = "local_libensemble"
                eval_str = "mpi_libensemble"
            else:
                eval_str = "mpi_comm_executor"
                # eval_str = "mpi_pool_executor"
            """
            input_space = createConfigSpace(queue, sk)
            print("TESTING YTOPT")
            max_evals = 500#5#50
            ytopt_tuning(queue, sk, platform_id, input_space, program_id=program_id, normalized_program_id=normalized_program_id,
                         max_flop_rate=max_flop_rate,
                         device_memory_bandwidth=device_memory_bandwidth,
                         device_latency=device_latency, timeout=timeout, save_path=save_path,
                         max_evals=max_evals, required_new_evals=max_evals, eval_str=eval_str)
        else:
            print("ONLY TESTING THE FIRST 20 transformations")
            from random import shuffle
            trans_list_list = einsum3to2_kernel_tlist_generator_v2(queue, sk)
            shuffle(trans_list_list)

            tdict = parallel_autotune(sk, platform_id, trans_list_list[:10], program_id=program_id,
                                      max_flop_rate=max_flop_rate, device_latency=device_latency,
                                      device_memory_bandwidth=device_memory_bandwidth, save_path=save_path,
                                      timeout=timeout)
    else:
        print("Not tuning", sk.default_entrypoint.name)
        # raise(ValueError(f"Unhandled einsum type: {est}"))

    # return list(trans_list_list[0])

    # Should this return the transformations or just save them?

    return True

# def autotune_dependent_subkernel(subkernel, queue):


def autotune_parts(parts, queue):
    from generators import einsum3to2_kernel_tlist_generator_v2, einsum2to2_kernel_tlist_generator_v2
    # from parallel_autotuning_charm4py_v2 import parallel_autotune
    from parallel_autotuning_mpi4py_v2 import parallel_autotune

    cum_transformations = []
    transformations = None
    counter = 0
    for csk, sk in parts:  # Subkernel and cumulative subkernel
        # Apply transformations to csk
        counter += 1

        # Determine einsum types of part and autotune based on those
        einsum_types = list(get_einsum_types(sk))
        if len(einsum_types) > 1:
            raise (ValueError(
                "Cannot currently handle multiple einsum types in same subkernel"))

        # for instr in sk.default_entrypoint.instructions:
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
            raise (ValueError("Unhandled einsum type"))

        # print(trans_list_list)
        # Test the kernel on the cumulative kernel
        # for entry in trans_list_list:
        #    for internal_entry in entry:
        #        print(internal_entry)
        #    print()
        # exit()
        # print(trans_list_list[0])
        # Just try the first transformation set for now
        cur_trans = cum_transformations + list(trans_list_list[0])

        # if counter == 2:
        # for entry in cur_trans:
        #    print(entry)
        # exit()

        # if counter != 2: # Avoid tuning the second part for the moment
        # autotune_result = run_single_param_set_v2(queue, csk, trans_list_list[1], generic_test)
        #    cum_transformations += list(autotune_result[1][:-1])

        # exit()
        # print(transformations)
        # exit()
        try:
            fs_csk = apply_feinsum_transformations(csk, queue)
            feinsum_tdict = run_single_param_set_v2(
                queue, fs_csk, [], generic_test)
            # tdict = parallel_autotune(csk, 0, trans_list_list)
            # print("Mine")
            # print(tdict)
            print("Feinsum")
            print(feinsum_tdict)

            exit()
        except RuntimeError:
            pass
        # transformations = tdict["transformations"]
        # Chop off redundant add_inames_for_unused_hw_axes
        # cum_transformations += trans_list_list[0] # Just use the first one for now

    # Save transformations to file (should probably also save the metadata)

    # print(cum_transformations)
    # exit()

    # return cum_transformations
    return transformations


def get_subkernels(tunit, args):

    barriers = get_barriers(tunit)
    if len(barriers) > 1:
        phases = get_phases(tunit, barriers)
        cum_subkernels = generate_cumulative_subkernels(
            tunit, barriers, phases)
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
    einsum_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions if any(
        [isinstance(tag, EinsumTag) for tag in instr.tags])}
    instr_dict = {
        instr.id: instr for instr in tunit.default_entrypoint.instructions}
    for esid, einsum in einsum_dict.items():
        es_deps = set(einsum.depends_on)
        while len(es_deps) > 0:
            dep_id = es_deps.pop()
            if dep_id in einsum_dict:
                return True
            else:
                es_deps |= instr_dict[dep_id].depends_on

    return False


def get_internal_einsum_dependencies(tunit):
    einsum_dict = {instr.id: instr for instr in tunit.default_entrypoint.instructions if any(
        [isinstance(tag, EinsumTag) for tag in instr.tags])}
    instr_dict = {
        instr.id: instr for instr in tunit.default_entrypoint.instructions}
    es_deps_dict = {}
    for esid, einsum in einsum_dict.items():
        deps = []
        es_deps = set(einsum.depends_on)
        while len(es_deps) > 0:
            dep_id = es_deps.pop()
            if dep_id in einsum_dict:
                deps.append(dep_id)
                es_deps |= instr_dict[dep_id].depends_on
        es_deps_dict[esid] = set(deps)
        # print(esid, "depends on the following einsums:", set(deps))
    return es_deps_dict


def get_pickled_tunits(directory_or_files):

    if isinstance(directory_or_files, str):
        files = os.listdir(directory_or_files)
        directory = directory_or_files
    else:
        # Assume it is a list of file names
        files = directory_or_files  # [ for file in directory_or_files]
        directory = None

    tunit_dicts = []

    for num, filename in list(enumerate(sorted(files))):
        if directory is not None:
            f = os.path.join(directory, filename)
        else:
            f = os.path.normpath(filename)

        _, filename = os.path.split(f)
        filename = str(filename)

        # Just looking at rank zero kernels for now.
        if os.path.isfile(f) and (filename.endswith(".pickle") or filename.endswith(".pkl")):

            if comm is None:  # POSIX file reading
                f = open(f, "rb")
                fdict = pickle.load(f)
                f.close()
            else:  # MPI IO
                #print("Beginning MPI IO")
                fsize = os.path.getsize(f)
                buf = bytearray(fsize)
                f = MPI.File.Open(comm, f)
                f.Read_all(buf)
                f.Close()
                fdict = pickle.loads(buf)
                #print("Ending MPI IO")

            if isinstance(fdict["tunit"], lp.TranslationUnit):
                tunit_dicts.append((filename, fdict,))

            # print(fdict["tunit"])

    return tunit_dicts


def get_lazy_einsum_info(tunit_dicts, hjson_dir=None):

    for filename, tunit_dict in tunit_dicts:
        tunit = tunit_dict["tunit"]
        print(tunit)
        args = tunit_dict["args"]
        sks = get_subkernels(tunit, args)
        # print(tunit.default_entrypoint)
        # contains_indirection(tunit)
        indirs = get_indirection_arrays(tunit)
        print(filename, len(sks), len(indirs) > 0)
        for sk, csk in sks:
            print(get_einsum_types(sk))
            print(sk.default_entrypoint.domains)
        # einsums = get_einsum_types(tunit)
        # for einsum in einsums:
        #    print("    ", einsum)
    # exit()

    # Count number of subkernels of each einsum type
    subkernel_counts = {}
    print("\nSubkernel information")
    pid_set = set()
    streaming_pid = set()

    einsum_0_to_0_pid = set()
    einsum_2_to_2_pid = set()
    einsum_3_to_3_pid = set()
    einsum_4_to_4_pid = set()
    einsum_5_to_5_pid = set()
    einsum_1_to_1_pid = set()
    einsum_3_to_2_pid = set()
    einsum_4_to_2_pid = set()
    einsum_5_to_3_pid = set()
    einsum_5_to_2_pid = set()
    einsum_2_to_1_pid = set()
    einsum_3_to_1_pid = set()
    non_einsum_pid = set()
    other_einsum_pid = set()

    einsum_0_to_0_batch_sizes = list()
    einsum_2_to_2_batch_sizes = list()
    einsum_3_to_3_batch_sizes = list()
    einsum_4_to_4_batch_sizes = list()
    einsum_5_to_5_batch_sizes = list()
    einsum_1_to_1_batch_sizes = list()
    einsum_3_to_2_batch_sizes = list()
    einsum_4_to_2_batch_sizes = list()
    einsum_5_to_3_batch_sizes = list()
    einsum_5_to_2_batch_sizes = list()
    einsum_2_to_1_batch_sizes = list()
    einsum_3_to_1_batch_sizes = list()
    other_einsum_batch_sizes = list()



    for filename, tunit_dict in tunit_dicts:

        # Restrict output to zero rank
        if ".pickle" in filename:
            tunit = tunit_dict["tunit"]
            args = tunit_dict["args"]
            sks = get_subkernels(tunit, args)
            # print("Number of subkernels", len(sks))

            for sk, csk in sks:

                # einsum_types = list(get_einsum_types(sk))
                einsum_counts = list(get_einsum_counts(sk).items())
                print("Einsum counts", einsum_counts)
                internal_deps = has_internal_einsum_dependencies(sk)
                # print_internal_einsum_dependencies(sk)

                indirection = len(get_indirection_arrays(sk)) > 0
                pid = unique_program_id(sk)
                pid_set |= {pid}

                # print(einsum_counts)
                if len(einsum_counts) > 1:
                    raise ValueError(
                        "There should not be multiple einsum types within a single subkernel")
                if len(einsum_counts) > 0:
                    einsum_type, count = einsum_counts[0]
                    non_red_axes = len(einsum_type[0])
                    red_axes = len(einsum_type[1])
                    total_axes = non_red_axes + red_axes
                    out_axes = total_axes - red_axes
                    key = (total_axes, out_axes, red_axes,
                           count, indirection, internal_deps)

                    # if total_axes == 5 and non_red_axes == 2:
                    #    print(sk)
                    #    exit()
                    if total_axes == 2 and non_red_axes == 2:
                        einsum_2_to_2_pid |= {pid}
                        einsum_2_to_2_batch_sizes.append(count)
                    elif total_axes == 3 and non_red_axes == 3:
                        einsum_3_to_3_pid |= {pid}
                        einsum_3_to_3_batch_sizes.append(count)
                    elif total_axes == 1 and non_red_axes == 1:
                        einsum_1_to_1_pid |= {pid}
                        einsum_1_to_1_batch_sizes.append(count)
                    elif total_axes == 4 and non_red_axes == 4:
                        einsum_4_to_4_pid |= {pid}
                        einsum_4_to_4_batch_sizes.append(count)
                    elif total_axes == 5 and non_red_axes == 5:
                        einsum_5_to_5_pid |= {pid}
                        einsum_5_to_5_batch_sizes.append(count)
                    elif total_axes == 0 and non_red_axes == 0:
                        einsum_0_to_0_pid |= {pid}
                        einsum_0_to_0_batch_sizes.append(count)
                    elif red_axes == 0:
                        print("Unclassified streaming:", total_axes, non_red_axes)
                        streaming_pid |= {pid}
                    elif total_axes == 3 and non_red_axes == 2:
                        einsum_3_to_2_pid |= {pid}
                        einsum_3_to_2_batch_sizes.append(count)
                    elif total_axes == 4 and non_red_axes == 2:
                        einsum_4_to_2_pid |= {pid}
                        einsum_4_to_2_batch_sizes.append(count)
                    elif total_axes == 5 and non_red_axes == 2:
                        einsum_5_to_2_pid |= {pid}
                        einsum_5_to_2_batch_sizes.append(count)
                    elif total_axes == 5 and non_red_axes == 3:
                        einsum_5_to_3_pid |= {pid}
                        einsum_5_to_3_batch_sizes.append(count)
                    elif total_axes == 2 and non_red_axes == 1:
                        einsum_2_to_1_pid |= {pid}
                        einsum_2_to_1_batch_sizes.append(count)
                        #print(sk)
                        #exit()
                    elif total_axes == 3 and non_red_axes == 1:
                        einsum_3_to_1_pid |= {pid}
                        einsum_3_to_1_batch_sizes.append(count)
                        #print(sk)
                        #exit()
                    else:
                        print("Unclassified: ", total_axes, non_red_axes, red_axes)
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
                        subkernel_counts[key][1] |= set(
                            [sk.default_entrypoint.name])
                    else:
                        subkernel_counts[key] = [
                            1, set([sk.default_entrypoint.name])]
                else:
                    non_einsum_pid |= {pid}

    print("Rank zero info")

    print("\nSubkernel summary information")
    for key, val in subkernel_counts.items():
        print(key, val)

    
    print("Number of distinct subkernels", len(pid_set))
    print("Number of distinct other streaming subkernels", len(streaming_pid))
    print("Non-einsum kernels", len(non_einsum_pid))

    labels = [
        "Number of distinct 0 to 0 einsums",
        "Number of distinct 1 to 1 einsums",
        "Number of distinct 2 to 2 einsums",
        "Number of distinct 3 to 3 einsums",
        "Number of distinct 4 to 4 einsums",
        "Number of distinct 5 to 5 einsums",
        "Number of distinct 2 to 1 einsums",
        "Number of distinct 3 to 1 einsums",
        "Number of distinct 3 to 2 einsums",
        "Number of distinct 4 to 2 einsums",
        "Number of distinct 5 to 2 einsums",
        "Number of distinct 5 to 3 einsums",
        "Number of distinct other einsums",
    ]

    pid_lens = [
        len(einsum_0_to_0_pid),
        len(einsum_1_to_1_pid),
        len(einsum_2_to_2_pid),
        len(einsum_3_to_3_pid),
        len(einsum_4_to_4_pid),
        len(einsum_5_to_5_pid),
        len(einsum_2_to_1_pid),
        len(einsum_3_to_1_pid),
        len(einsum_3_to_2_pid),
        len(einsum_4_to_2_pid),
        len(einsum_5_to_2_pid),
        len(einsum_5_to_3_pid),
        len(other_einsum_pid),
    ]

    batch_size_counts = [
        einsum_0_to_0_batch_sizes,
        einsum_1_to_1_batch_sizes,
        einsum_2_to_2_batch_sizes,
        einsum_3_to_3_batch_sizes,
        einsum_4_to_4_batch_sizes,
        einsum_5_to_5_batch_sizes,
        einsum_2_to_1_batch_sizes,
        einsum_3_to_1_batch_sizes,
        einsum_3_to_2_batch_sizes,
        einsum_4_to_2_batch_sizes,
        einsum_5_to_2_batch_sizes,
        einsum_5_to_3_batch_sizes,
        other_einsum_batch_sizes,
    ]

    from scipy.stats import mode
    for label, pid_len, batch_size_list in zip(labels, pid_lens, batch_size_counts):
        try:
            print(label, pid_len, np.min(batch_size_list), np.max(batch_size_list), np.median(batch_size_list), np.mean(batch_size_list), mode(batch_size_list))
        except Exception:
            print(label, pid_len)
    """
    print("Number of distinct 1 to 1 einsums", len(einsum_1_to_1_pid))
    print("Number of distinct 2 to 2 einsums", len(einsum_2_to_2_pid))
    print("Number of distinct 3 to 3 einsums", len(einsum_3_to_3_pid))
    print("Number of distinct 4 to 4 einsums", len(einsum_4_to_4_pid))
    print("Number of distinct 5 to 5 einsums", len(einsum_5_to_5_pid))
    print("Number of distinct 2 to 1 einsums", len(einsum_2_to_1_pid))
    print("Number of distinct 3 to 1 einsums", len(einsum_3_to_1_pid))
    print("Number of distinct 3 to 2 einsums", len(einsum_3_to_2_pid))
    print("Number of distinct 4 to 2 einsums", len(einsum_4_to_2_pid))
    print("Number of distinct 5 to 2 einsums", len(einsum_5_to_2_pid))
    print("Number of distinct 5 to 3 einsums", len(einsum_5_to_3_pid))
    print("Number of distinct other einsums", len(other_einsum_pid))
    """

def get_device_roofline_data(queue):
    import feintune.empirical_roofline as er
    results_list = er.loopy_bandwidth_test(
        queue, fast=True, print_results=True, fill_on_device=True)
    device_latency = er.get_min_device_latency(results_list)
    loopy_bw = er.get_latency_adjusted_max_device_memory_bandwidth(
        results_list)
    clpeak_bw = er.get_max_bandwidth_clpeak(queue=queue)
    clpeak_flop_rate = er.get_max_flop_rate_clpeak(np.float64, queue=queue)
    device_memory_bandwidth = max(loopy_bw, clpeak_bw)

    return device_latency, device_memory_bandwidth, clpeak_flop_rate


def autotune_standalone_subkernels(queue, sk_list, save_path=None, device_latency=None, device_memory_bandwidth=None, peak_flop_rate=None):

    print("AUTOTUNING STANDALONE SUBKERNELS")

    #platforms = cl.get_platforms()
    #cl_ctx = cl.Context(
    #    dev_type=cl.device_type.GPU,
    #    properties=[(cl.context_properties.PLATFORM, platforms[0])])
    #queue = cl.CommandQueue(cl_ctx,
    #                        properties=cl.command_queue_properties.PROFILING_ENABLE)

    if save_path is None:
        save_path = "./hjson"

    # TODO: Cache the roofline result or pass it in as an argument.
    """
    if False:
        if not use_charm:
            if comm.Get_rank() == 0:
                # The latency is now obtained per-kernel so it probably needn't be obtained here.
                # Bandwidth microbenchmark could be improved to handle asymmetric numbers of reads and
                # writes.
                device_latency, device_memory_bandwidth, peak_flop_rate = get_device_roofline_data(queue)
            else:
                device_memory_bandwidth = None
                device_latency = None
                peak_flop_rate = None

            device_memory_bandwidth = comm.bcast(device_memory_bandwidth)
            device_latency = comm.bcast(device_latency)
            clpeak_flop_rate = comm.bcast(clpeak_flop_rate)
        else:
            device_latency, device_memory_bandwidth, peak_flop_rate = get_device_roofline_data(queue)
    else:
        device_memory_bandwidth = None
        device_latency = None
        peak_flop_rate = None
    """

    for sk_dict in sk_list:
        pid = sk_dict["pid"]
        npid = sk_dict["npid"]
        sk = sk_dict["sk"]

        if False:  # Feinsum autotuning
            feinsum_autotune(tunit, queue)
        else:  # Eager-style autotuning

            os.makedirs(save_path, exist_ok=True)
            hjson_file = f"{save_path}/{pid}.hjson"
            if exists(hjson_file):
                print(f"A TUNE PROFILE ALREADY EXISTS: {hjson_file}")
            else:
                print(f"A TUNE PROFILE EXISTS NOT: {hjson_file}")

            if True:
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

                    print("EINSUM INFO:", total_axes, non_red_axes,
                          red_axes, indirection, einsum_count, pid, npid)

                    handled_pairs = set([(2, 1,), (3, 2,), (2, 2,), (2, 3)])
                    if (non_red_axes, red_axes,) in handled_pairs and not indirection and einsum_count <= 10:
                        # Add indirection arrays as a parameter?
                        autotune_standalone_subkernel(sk, queue, program_id=pid, normalized_program_id=npid,
                                                      max_flop_rate=peak_flop_rate,
                                                      device_latency=device_latency,
                                                      device_memory_bandwidth=device_memory_bandwidth,
                                                      save_path=save_path)


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
            if len(get_indirection_arrays(tunit)) == 0:
                try:
                    apply_feinsum_transformations(sk, queue)
                except RuntimeError:
                    print("Couldn't find transformation for", filename)


def compare_weighted_avg_frac_rooflines(directory, pid_dict):
    tuned_dir = directory + "/hjson"
    untuned_dir = directory + "/default_transforms_hjson"

    tuned_files = os.listdir(tuned_dir)
    untuned_files = os.listdir(untuned_dir)

    overlapping_files = set(tuned_files) & set(untuned_files)

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
            # print(f)
            dct = load_hjson(f)
            data.append((pid, dct["data"],))
            total_avg_exec_time += pid_dict[pid]*(
                dct["data"]["avg_time"] - dct["data"]["device_memory_latency"])
            # total_avg_exec_time += dct["data"]["avg_time"] - dct["data"]["device_memory_latency"]

        weighted_avg_roofline = 0
        for pid, entry in data:
            weighted_avg_roofline += pid_dict[pid]*entry["frac_roofline_flop_rate"]*(
                entry["avg_time"] - entry["device_memory_latency"])/total_avg_exec_time
            # weighted_avg_roofline += entry["frac_roofline_flop_rate"]*(entry["avg_time"] - entry["device_memory_latency"])/total_avg_exec_time

        return weighted_avg_roofline

    print(overlapping_files)
    tuned_frac_roofline = get_data(overlapping_files, tuned_dir, pid_dict)
    untuned_frac_roofline = get_data(overlapping_files, untuned_dir, pid_dict)

    print(len(overlapping_files), len(untuned_files),
          untuned_frac_roofline, tuned_frac_roofline)


def collect_subkernels(tunit_dicts):

    out_list = []
    pid_counts = {}
    for filename, fdict in tunit_dicts:
        tunit = fdict["tunit"]
        args = fdict["args"]
        print(f"OBTAINING SUBKERNELS FROM: {filename}")
        #print(tunit)
        sks = get_subkernels(tunit, args)

        for sk, csk in sks:
            # This may change the identifier so needs to be set beforehand
            # assert sk.default_entrypoint.options.no_numpy
            # assert sk.default_entrypoint.options.return_dict
            pid = unique_program_id(sk, attempt_normalization=False)
            npid = unique_program_id(sk, attempt_normalization=True)
            # out_list.append((pid, sk, csk,))
            out_list.append({"pid": pid, "npid": npid, "sk": sk, "csk": csk})

            # Could also do this with Collections.Counter
            if pid in pid_counts:
                pid_counts[pid] += 1
            else:
                pid_counts[pid] = 1

    return out_list, pid_counts


# TODO. Make load and save paths arguments
def main(args):

    # dump_subkernels_from_pickled(None)
    # directory = "./pickled_programs_prediction"
    directories = [args.indir#"./pickled_programs"
                   # "./pickled_programs_y3_prediction_order_1_eager",
                   # "../pickled_programs_y3_prediction_order_2_lazy",
                   # "./pickled_programs_wave",
                   # "./pickled_programs_prediction_order_1",
                   # "./pickled_programs_y3_prediction_order_1",
                   # "./pickled_programs_y3_prediction_order_3",
                   # "./pickled_programs_prediction_order_2",
                   # "./pickled_programs_prediction_order_3",
                   # "./pickled_programs_prediction_order_4"
                   ]

    # Could sort subkernels by dimensions, then use the maximum long axis
    # for the kernels that share the other dimensions as the dimension
    # length for tuning purposes. Then map the subkernels keys
    # to that hjson file (or just duplicate the hjson file for each subkernel
    # key). Need to count that the number of inames is the same though. And need to
    # figure out the element iname

    # Or just have the size be a parameter in the Bayesian optimization space.

    # platforms = cl.get_platforms()
    # cl_ctx = cl.Context(
    #    dev_type=cl.device_type.GPU,
    #    properties=[(cl.context_properties.PLATFORM, platforms[0])])
    # queue = cl.CommandQueue(cl_ctx,
    #    properties=cl.command_queue_properties.PROFILING_ENABLE)
    platforms = cl.get_platforms()
    pnum_saved = 0
    for pnum, platform in enumerate(platforms):
        if platform.vendor == "The pocl project":#"NVIDIA Corporation":
            pnum_saved = pnum
    for pnum, platform in enumerate(platforms):
        if platform.vendor == "Advanced Micro Devices, Inc.":#"NVIDIA Corporation":
            pnum_saved = pnum



    devices = platforms[pnum_saved].get_devices(device_type=cl.device_type.GPU)
    if comm is not None:
        cl_ctx = cl.Context(
            devices=[devices[comm.Get_rank() % len(devices)]])
        # properties=[(cl.context_properties.PLATFORM, platforms[0])])
    else:
        cl_ctx = cl.Context(devices=devices)
    queue = cl.CommandQueue(cl_ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    device_latency = None
    device_memory_bandwidth = None
    clpeak_flop_rate = None
    print("BENCHMARK", args.benchmark)
    # Need to handle oversubscribed GPU by profiling in waves or by communicating results.
    # Actually, communicating is probably the better way to do this. Just have the first
    # rank profile.
    if args.benchmark:
        if comm is None or (comm is not None and comm.Get_rank() == 0):
            device_latency, device_memory_bandwidth, clpeak_flop_rate = get_device_roofline_data(
                queue)

    if comm is not None:
        comm.Barrier()

    from meshmode.array_context import PrefusedFusionContractorArrayContext
    actx = PrefusedFusionContractorArrayContext(queue)

    for directory in directories:
        save_path = args.outdir#"./autotuning_files"  # directory + "/hjson3"
        # Really a tuple, not a dict
        tunit_dicts = get_pickled_tunits(directory)
        tunit_dicts = sorted(tunit_dicts, key=lambda entry: get_knl_flops(entry[1]["tunit"]), reverse=True)
        tunit_dicts = tunit_dicts[:50]

        #print(tunit_dicts[0][1]["tunit"])
        #exit()
        #for entry in tunit_dicts:
        #    print(get_knl_flops(entry[1]["tunit"]))
        #exit()

        if False:  # Tune a single macrokernel at a time.

            for tunit_dict in tunit_dicts:

                transformed_tunit, transformed_subkernels = transform_macrokernel(tunit_dict, save_path, in_actx=None, tune=False,
                                                                                  device_latency=device_latency, 
                                                                                  device_memory_bandwidth=device_memory_bandwidth, 
                                                                                  peak_flop_rate=clpeak_flop_rate)
                #transformed_tunit_default = actx.transform_loopy_program(tunit_dict[1]["tunit"])
                transformed_tunit_default, transformed_subkernels_default = transform_macrokernel(tunit_dict, save_path + "_default", in_actx=actx, tune=False)

                # test_kernels(transformed_subkernels_default, queue, save_path=None, device_latency=device_latency, device_memory_bandwidth=device_memory_bandwidth, peak_flop_rate=clpeak_flop_rate)

                # Would need to save the indirection arrays with the kernel
                if False:#len(get_indirection_arrays(transformed_tunit)) == 0: #any([arg.dtype.dtype == np.int8 for arg in transformed_tunit.default_entrypoint.args]):
                    ret_dict1 = run_single_param_set_v2(queue, transformed_tunit, [], generic_test,
                                max_flop_rate=clpeak_flop_rate, device_memory_bandwidth=device_memory_bandwidth,
                                device_latency=device_latency)
                    #print(ret_dict)
                    #print("Combined - Transformed time:", ret_dict1["data"]["avg_time"]) 

                    #print(transformed_tunit_default)
                    ret_dict2 = run_single_param_set_v2(queue, transformed_tunit_default, [], generic_test,
                                max_flop_rate=clpeak_flop_rate, device_memory_bandwidth=device_memory_bandwidth,
                                device_latency=device_latency)
                    print("Combined - Default time:", ret_dict2["data"]["avg_time"], "Combined - Transformed time:", ret_dict1["data"]["avg_time"])
                    #exit()

        if True:  # Tune all of the subkernels
            from feintune.utils import tunit_to_einsum
            print("Done collecting tunits")
            # ID changes based on whether python was run with -O
            sk_list, pid_dict = collect_subkernels(tunit_dicts)
            sk_list = sorted(sk_list, key=lambda e: get_knl_flops(
                e["sk"]), reverse=True)#[20:21]#[112:]
            #"""
            #sk_list = sorted(sk_list, key=lambda e: e["sk"].default_entrypoint.name)
            #for item in sk_list:
            #    print(item["sk"].default_entrypoint.name)

            if False:
                for item in sk_list[:]:
                    sk = item["sk"]
                    print(sk)
                    print(item["pid"], item["npid"])
                    try:
                        if len(get_indirection_arrays(sk)) == 0:
                            einsum = tunit_to_einsum(sk)
                            print(einsum)
                    except NotImplementedError as e:
                        print(e)
                    #except RuntimeError as e:
                    #    print("RUNTIME ERROR")
                    #    print(sk)
                    #    print(e)
                    except ValueError as e:
                        print("VALUE ERROR")
                        #print(e)
                    #except AttributeError as e:
                        # What is this aggregate attribute?
                    #    print("ATTRIBUTE ERROR")
                    #print(unique_program_id(sk))
                    #if unique_program_id(sk) in {"2a82b7f82159384d828d2b94704327f0fcf46209ad6a43bcc9842b43beeb56c2",
                    #                             "9701d4c523fff28c6a3d78b294f1cfc8f5766aca1380780e340bdba4d4a3a863",
                    #                             "d5ef78e056a3aa17951ef94b683f8a3a683ddc7eda0d9cb8ea20c754a6533e9d",
                    #                             "f3dbebb372a1f5e2e7002640108747dd3274fab9d9f89f50ae1581fe482871fb"}:
                    #    einsum = tunit_to_einsum(sk)
                    #    exit()
                #exit()
                #"""
                #exit()
            # sk_list = [tunit_dict[1]["tunit"] for tunit_dict in tunit_dicts]
            # """
            # sk_list = [sk for _, sk, _ in sk_list]
            # for sk in sk_list:
            #    sk_to_print = ["unfiltered_rhs_20"]
            #    if sk.default_entrypoint.name in sk_to_print:
            #        print(sk)
            # exit()
            # """
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
            #exit()
            """
            print("Done collecting subkernels")
            #get_lazy_einsum_info(tunit_dicts, hjson_dir=save_path)
            #exit()

            #test_default_transforms(sk_list, save_path=directory + "/default_transforms_hjson")
            autotune_standalone_subkernels(queue, sk_list, save_path=save_path, 
                                           device_latency=device_latency,
                                           device_memory_bandwidth=device_memory_bandwidth, peak_flop_rate=clpeak_flop_rate)
            # compare_weighted_avg_frac_rooflines(directory, pid_dict)

    exit()


if __name__ == "__main__":
    if use_charm:
        charm.start(main)
        exit()  # charm.exit freezes the program
        charm.exit()
    else:
        main(0)
