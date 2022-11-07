import hjson

def load_hjson(filename):
    hjson_file = open(filename, "rt")
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od

def convert(o):
    from numpy import generic, inf, finfo, float32
    from frozendict import frozendict
    if o == inf:
        return finfo(float32).max
    elif isinstance(o, generic): 
        return o.item()
    elif isinstance(o, frozendict):
        return dict(o)
    raise TypeError

def dump_hjson(filename, out_dict): 
    out_file = open(filename, "wt")
    hjson.dump(out_dict, out_file, default=convert)
    out_file.close()

def unique_program_id(tunit):
    from loopy.tools import LoopyKeyBuilder
    kb = LoopyKeyBuilder()
    # The program name is not relevant for transformation purposes. 
    # (Neither are the variable names, but I'm not going to touch that)
    # Maybe feinsum has some capability for that?
    assert len(tunit.entrypoints) == 1 # Only works for tunits with one entrypoint at present

    return kb(tunit.default_entrypoint.copy(name="loopy_kernel"))
