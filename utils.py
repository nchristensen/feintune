from hashlib import md5
import numpy as np
from frozendict import frozendict

def convert(o):
    if isinstance(o, np.generic): 
        return o.item()
    elif isinstance(o, frozendict):
        return dict(o)
    raise TypeError

def unique_program_id(program):
    #code = lp.generate_code_v2(program).device_code() # Not unique
    #return md5(str(program.default_entrypoint).encode()).hexdigest() # Also not unique

    ep = program.default_entrypoint
    domains = ep.domains
    instr = [str(entry) for entry in ep.instructions]
    args = ep.args
    #name = ep.name # Should look at temporaries, also the name is possibly irrelevant

    # Is the name really relevant? 
    #all_list = [name] + domains + instr + args
    # Somehow this can change even if the string is the same
    #identifier = md5(str(all_list).encode()).hexdigest()

    """
    print("NAME")
    print(name)
    print()
    print("DOMAINS")
    print(domains)
    print()
    print("INSTRUCTIONS")
    print(instr)
    print()
    print("ARGS")
    print(args)
    print()
    """

    # Can sorting the instructions change the meaning of the loopy program?
    dstr = md5(str(sorted(domains)).encode()).hexdigest() #List
    istr = md5(str(sorted(instr)).encode()).hexdigest()   #List
    astr = md5(str(sorted(args, key=lambda arg: arg.name))).encode()).hexdigest()    #List
    #nstr = md5(name.encode()).hexdigest()
    #print("dstr", dstr)
    #print("nstr", nstr)
    #print("istr", istr)
    #print("astr", astr)
    #for entry in all_list:
    #    print(entry)
    #print(str(all_list))
    #identifier = nstr[:4] + dstr[:4] + istr[:4] + astr[:4]

    identifier = dstr[:4] + istr[:4] + astr[:4]

    return identifier

