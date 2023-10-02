import hjson
import loopy as lp
from loopy.symbolic import CombineMapper, DependencyMapper
from typing import FrozenSet
import numpy as np
from pytools import memoize
from frozendict import frozendict

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
            canonical_einsum = f.canonicalize_einsum(f.get_a_matched_einsum(tunit))
            key = kb(canonical_einsum)
            #print("Successfully normalized einsum")
            #print(canonical_einsum)
        except Exception:
            #print("Failed to normalize tunit, using non-normalized program_id.")
            key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))
    else:
        key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))

    key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))

    return key


# Map a tuple of inames to a domain
@memoize
def get_domain_list(tunit):
    domains = tunit.default_entrypoint.domains
    domain_list = []
    for domain in domains:
        #print(domain.get_var_names(islpy.dim_type.all))
        domain_names = frozenset({key.name for key in domain.get_id_dict().keys()})
        domain_list.append((domain_names, domain,))

    #import islpy
    #for domain_names, domain in domain_list:
    #    print(domain_names, domain)
    #    id_dict = domain.get_id_dict()
    #    print(id_dict)
    #    exit()
    #    for dim in range(domain.n_dim()):
    #        print(domain.dim_max(dim))
            
    #    #print(domain.drop_constraints_involving_dims(islpy.dim_type.all, 0, 1))
    #    exit()

    return tuple(domain_list)

#Map an iname to a tuple of its min and max value assuming
# these are constants
from loopy.translation_unit import for_each_kernel

@memoize
def get_iname_limits(knl):
    domains = knl.domains
    iname_limits = {}
    for domain in domains:
        id_dict = domain.get_id_dict()
        for key, tup in id_dict.items():
            #max_val = int(str(domain.dim_max_val(index)))
            #min_val = int(str(domain.dim_min_val(index)))
            # Not sure we need the actual values at this point.
            max_val = domain.dim_max_val(tup[1])
            min_val = domain.dim_min_val(tup[1])
            limits = (min_val, max_val,)
            # Assume the limits are consistent within a kernel.
            if key.name in iname_limits:
                assert iname_limits[key.name] == limits
            iname_limits[key.name] = (min_val, max_val,)
    return frozendict(iname_limits)
            

## Kaushik's indirection finder code

"""
class IndirectionFinder(CombineMapper):
    def __init__(self, all_inames: FrozenSet[str]):
        super().__init__()
        self.all_inames = all_inames

    def combine(self, values):
        return any(values)

    def map_subscript(self, expr):
        import pymbolic.primitives as prim
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

# Some test code
if False:
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

