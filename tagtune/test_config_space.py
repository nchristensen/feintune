from ConfigSpace.read_and_write import json as cs_json
from dataclasses import dataclass
import ConfigSpace as cs
from ConfigSpace import *

a = cs.OrdinalHyperparameter("a", [1])
b = cs.OrdinalHyperparameter("b", [2])

# Uniformly distributed
a = cs.Integer("a", (1, 10))
a = cs.Integer("a", (1, 10), distribution=cs.Uniform())

# Normally distributed at 2 with std 3
b = cs.Integer("b", distribution=Normal(2.5, 3))
b = cs.Integer("b", (0, 5), distribution=cs.Normal(2.5, 3.5))  # ... bounded

cspace = cs.ConfigurationSpace()
cspace.add_hyperparameter(a)
cspace.add_hyperparameter(b)

# print(cspace["a"] < cspace["b"])


@dataclass
class ProductGreaterThan:
    limit: int

    def __call__(self, a, b):
        return a*b > self.limit


forbid = ProductGreaterThan(25)
print(forbid.__class__.__qualname__)

a_times_b_greater_than_25 = cs.ForbiddenCallableRelation(a, b, forbid)
cspace.add_forbidden_clause(a_times_b_greater_than_25)

print(cspace)

jsn = cs_json.write(cspace, pickle_callables=True)
print(jsn)
with open('configspace.json', 'w') as f:
    f.write(jsn)

with open('configspace.json', 'r') as f:
    json_string = f.read()
    print(json_string)
    cspace_loaded = cs_json.read(json_string)

print(cspace_loaded)

exit()
