def convert(o):
    from numpy import generic
    from frozendict import frozendict
    if isinstance(o, generic): 
        return o.item()
    elif isinstance(o, frozendict):
        return dict(o)
    raise TypeError

def unique_program_id(tunit):
    from loopy.tools import LoopyKeyBuilder
    kb = LoopyKeyBuilder()
    # The program name is not relevant for transformation purposes. 
    # (Neither are the variable names, but I'm not going to touch that)
    assert len(tunit.entrypoints) == 1 # Only works for tunits with one entrypoint at present
    return kb(tunit.default_entrypoint.copy(name="loopy_kernel"))
