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

def unique_program_id(tunit, attempt_normalization=True):
    from loopy.tools import LoopyKeyBuilder
    kb = LoopyKeyBuilder()

    assert len(tunit.entrypoints) == 1 # Only works for tunits with one entrypoint at present

    # The program name is not relevant for transformation purposes. 
    # (Neither are the variable names, but I'm not going to touch that)
    # Maybe feinsum has some capability for that?
    
    # Kernel may not necessarily be an einsum, but for now assume it is
    # (the tuner also doesn't care if there are einsums with different loop
    # dimensions in the same kernel
    if attempt_normalization:
        import feinsum as f
        try:
            # Not every einsum can currently be normalized, for instance
            # if it has a non-reduction RHS or if it has indirection
            canonical_einsum = f.normalize_einsum(f.match_einsum(tunit))
            key = kb(canonical_einsum)
            print("Successfully normalized einsum")
            print(canonical_einsum)
        except Exception:
            print("Failed to normalize tunit, using non-normalized program_id.")
            key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))
    else:
        key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))

    return key
