import pickle
import loopy as lp
from pytools.tag import Tag
from meshmode.array_context import EinsumTag

class IsDOFArray(Tag):
    pass


file_path = "./pickled_programs/03dccf17ebb345c3.pickle"
f = open(file_path, "rb")
tunit = pickle.load(f)


f.close()

print(tunit)

### Apply tags

# Just slap the tag on the arrays for now. Will eventually need to figure out how to propagate the tags
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

### End tag application

#print(tunit.default_entrypoint.domains)
#print(tunit.default_entrypoint.args)
#print(tunit.default_entrypoint.temporary_variables)

# Map a domain to a tuple for its inames
domains = tunit.default_entrypoint.domains
import islpy
domain_list = []
for domain in domains:
    #print(domain.get_var_names(islpy.dim_type.all))
    domain_names = frozenset([key.name for key in domain.get_id_dict().keys()])
    domain_list.append((domain_names, domain,))
    print(domain_names)

# Get the barriers to divide computation into phases

barriers = [None]
for instr in tunit.default_entrypoint.instructions:

    #if any([isinstance(tag, EinsumTag) for tag in instr.tags]):
    #    print(str(instr))
    if isinstance(instr, lp.BarrierInstruction) and instr.synchronization_kind == "global":
        barriers.append(instr.id)
        print(str(instr.id))

# Get the barriers to divide computation into phases
phase_lists = [{"domains": frozenset(), "instructions": [], "args": frozenset()} for i in range(len(barriers) + 1)]
phases = dict(zip(barriers, phase_lists))
print(phases)

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


    print(phases[dbarrier]["domains"])

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


# Temporaries may need to be privatized again here

# See if we can create a subkernel with the domains and instructions of each cumulative phase
for cur_phase in range(len(barriers[0:3])):
    print(f"BARRIER {barriers[cur_phase]}")
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
    print(knl)

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




