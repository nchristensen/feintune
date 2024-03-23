from loopy.translation_unit import for_each_kernel
import hjson
import loopy as lp
from loopy.symbolic import CombineMapper, DependencyMapper
from typing import FrozenSet
import numpy as np
from pytools import memoize
from immutabledict import immutabledict


def mpi_read_all(filename):
    import mpi4py.MPI as MPI
    import os
    import io
    #print("Beginning MPI IO")
    fsize = os.path.getsize(filename)
    buf = bytearray(fsize)
    f = MPI.File.Open(comm, filename)
    f.Read_all(buf)
    f.Close()
    file = io.TextIOWrapper(buf)
    return file
    #print("Ending MPI IO")


def load_hjson(filename):
    hjson_file = open(filename, "rt")
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od


def convert(o):
    from numpy import generic, inf, finfo, float32
    from immutabledict import immutabledict
    if o == inf:
        return finfo(float32).max
    elif isinstance(o, generic):
        return o.item()
    elif isinstance(o, immutabledict):
        return dict(o)
    raise TypeError


def dump_hjson(filename, out_dict):
    out_file = open(filename, "wt")
    hjson.dump(out_dict, out_file, default=convert)
    out_file.close()

# Copied from meshmode
def tunit_to_einsum(t_unit):
        from arraycontext.impl.pytato.compile import FromArrayContextCompile
        from functools import reduce

        #if t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):
        # FIXME: Enable this branch, WIP for now and hence disabled it.
        from loopy.match import ObjTagged
        import feinsum as fnsm
        from meshmode.array_context import _get_elementwise_einsum, EinsumTag
        #from meshmode.feinsum_transformations import FEINSUM_TO_TRANSFORMS

        einsum_tags = reduce(
            frozenset.union,
            (insn.tags_of_type(EinsumTag)
             for insn in t_unit.default_entrypoint.instructions),
            frozenset())

        if len(einsum_tags) > 0 and all(insn.tags_of_type(EinsumTag)
                   for insn in t_unit.default_entrypoint.instructions
                   if isinstance(insn, lp.MultiAssignmentBase)
                   ) and len(get_indirection_arrays(t_unit)) == 0:# and not " if " in str(t_unit):

            #print(t_unit)
            #tunit_str = str(t_unit)
            #if ("tan" in tunit_str or "pow" in tunit_str or "sin" in tunit_str) and not ("subst_0" in tunit_str):
            #    exit()
            #for insn in t_unit.default_entrypoint.instructions:
            #    for tag in insn.tags:
            #        print(type(tag))
            #assert all(insn.tags_of_type(EinsumTag)
            #           for insn in t_unit.default_entrypoint.instructions
            #           if isinstance(insn, lp.MultiAssignmentBase)
            #           )

            #print(len(einsum_tags))
            #if len(einsum_tags) == 0:
            #    exit()
            #assert len(einsum_tags) <= 1
            #print("Indirection arrays:", get_indirection_arrays(t_unit))
            #norm_fused_einsum = None

            for ensm_tag in sorted(einsum_tags,
                                   key=lambda x: sorted(x.orig_loop_nest)):
                if reduce(frozenset.union,
                          (insn.reduction_inames()
                           for insn in (t_unit.default_entrypoint.instructions)
                           if ensm_tag in insn.tags),
                          frozenset()):
                    #try:
                    fused_einsum = fnsm.match_einsum(t_unit, ObjTagged(ensm_tag))
                    #except ValueError:
                        #print("Could not match einsum")
                        #print(t_unit)
                        #return

                    # New version, appears to be broken.
                    #fused_einsum, subst_map = fnsm.get_a_matched_einsum(t_unit, insn_match=ObjTagged(ensm_tag))
                    #print("MATCHED THE EINSUMS")
                else:
                    # elementwise loop
                    fused_einsum = _get_elementwise_einsum(t_unit, ensm_tag)
                #print("canonicalizing")
                #print(fused_einsum)
                norm_fused_einsum = fnsm.normalize_einsum(fused_einsum)
                #norm_fused_einsum = fnsm.canonicalize_einsum(fused_einsum)
                #print("printing")
                import hashlib
                print(str(norm_fused_einsum))
                h = hashlib.md5(str(norm_fused_einsum).encode('utf-8')).hexdigest()
                print(h)
                print(t_unit)
                #exit()

            return norm_fused_einsum

            """
            try:
                fnsm_transform = FEINSUM_TO_TRANSFORMS[
                    fnsm.normalize_einsum(fused_einsum)]
            except KeyError:
                fnsm.query(fused_einsum,
                           self.queue.context,
                           err_if_no_results=True)
                1/0

            t_unit = fnsm_transform(t_unit,
                                    insn_match=ObjTagged(ensm_tag))
            """

            #else:
            #    print(t_unit)
            #    raise RuntimeError
        else:
            raise NotImplementedError
            #print("No einsum tags found")


def unique_program_id(tunit, attempt_normalization=True):
    from loopy.tools import LoopyKeyBuilder
    kb = LoopyKeyBuilder()

    # Only works for tunits with one entrypoint at present
    assert len(tunit.entrypoints) == 1

    # The program name is not relevant for transformation purposes.
    # (Neither are the variable names, but I'm not going to touch that)
    # Maybe feinsum has some capability for that?

    # Kernel may not necessarily be an einsum, but for now assume it is
    # (the tuner also doesn't care if there are einsums with different loop
    # dimensions in the same kernel
    from feintune.apply_transformations import get_einsum_types
    et = list(get_einsum_types(tunit))
    neinsums = len(et)
    if neinsums > 0:
        nreduction = len(et[0][1])


    if attempt_normalization and neinsums > 0:# and nreduction > 0:
        import feinsum as f
        try:
            # Not every einsum can currently be normalized, for instance
            # if it has a non-reduction RHS or if it has indirection
            #print(tunit)
            #print("nreduction", nreduction)
            #print("Substitutions:", tunit.default_entrypoint.substitutions)
            #if nreduction > 0:
            #arg_subs = None#frozenset([arg.name for arg in tunit.default_entrypoint.args])
            #print("arg_subs", arg_subs)
            #matched_einsum = f.get_a_matched_einsum(tunit, argument_substitutions=arg_subs)
            #canonical_einsum = f.canonicalize_einsum(matched_einsum)
            normalized_einsum = tunit_to_einsum(tunit)
            import hashlib
            #print(str(norm_fused_einsum))
            key = hashlib.md5(str(normalized_einsum).encode('utf-8')).hexdigest()
            #print(key)
            #exit() 
            #key = kb(canonical_einsum)
            # print("Successfully normalized einsum")
            # print(canonical_einsum)
        except Exception:
            print("Failed to normalize tunit, using non-normalized program_id.")
            key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))
    else:
        key = kb(tunit.default_entrypoint.copy(name="loopy_kernel"))


    return key


# Map a tuple of inames to a domain
@memoize
def get_domain_list(tunit):
    domains = tunit.default_entrypoint.domains
    domain_list = []
    for domain in domains:
        # print(domain.get_var_names(islpy.dim_type.all))
        domain_names = frozenset(
            {key.name for key in domain.get_id_dict().keys()})
        domain_list.append((domain_names, domain,))

    # import islpy
    # for domain_names, domain in domain_list:
    #    print(domain_names, domain)
    #    id_dict = domain.get_id_dict()
    #    print(id_dict)
    #    exit()
    #    for dim in range(domain.n_dim()):
    #        print(domain.dim_max(dim))

    #    #print(domain.drop_constraints_involving_dims(islpy.dim_type.all, 0, 1))
    #    exit()

    return tuple(domain_list)


# Map an iname to a tuple of its min and max value assuming
# these are constants


@memoize
def get_iname_limits(knl):
    domains = knl.domains
    iname_limits = {}
    for domain in domains:
        id_dict = domain.get_id_dict()
        for key, tup in id_dict.items():
            # max_val = int(str(domain.dim_max_val(index)))
            # min_val = int(str(domain.dim_min_val(index)))
            # Not sure we need the actual values at this point.
            max_val = domain.dim_max_val(tup[1])
            min_val = domain.dim_min_val(tup[1])
            limits = (min_val, max_val,)
            # Assume the limits are consistent within a kernel.
            if key.name in iname_limits:
                assert iname_limits[key.name] == limits
            iname_limits[key.name] = (min_val, max_val,)
    return immutabledict(iname_limits)


# Kaushik's indirection finder code

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

        # print(expr, expr.aggregate)
        # print(super_subscript, aggregate)

        # print(super_subscript, super_subscript - aggregate) #not should_record else frozenset()
        if not should_record:
            retval = super_subscript - aggregate  # not should_record else frozenset()
        else:
            retval = super_subscript
        # print(retval)
        return retval

    def map_variable(self, expr, should_record=False):
        # print("MAP VARIABLE", should_record)
        return super().map_variable(expr, should_record=should_record) if should_record else frozenset()

    # def map_constant(self, expr, should_record=False):
        # print("MAP CONSTANT", expr)
        # return frozenset()


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
    # print("RHS index deps are:", result)
    retval = frozenset([var.name for var in result])
    return retval


def get_indirection_arrays(tunit):
    index_deps = get_index_deps(tunit)
    inames = frozenset(tunit.default_entrypoint.inames.keys())
    indirection_arrays = index_deps - (index_deps & inames)
    # print("Indirection arrays:", indirection_arrays)
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
