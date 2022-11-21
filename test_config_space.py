import ConfigSpace as cs
from ConfigSpace import *

a = cs.OrdinalHyperparameter("a", [1])
b = cs.OrdinalHyperparameter("b", [2])

# Uniformly distributed
a = cs.Integer("a", (1, 10))
a = cs.Integer("a", (1, 10), distribution=cs.Uniform())

# Normally distributed at 2 with std 3
b = cs.Integer("b", distribution=Normal(2, 3))
b = cs.Integer("b", (0, 5), distribution=cs.Normal(2, 3))  # ... bounded

cspace = cs.ConfigurationSpace()
cspace.add_hyperparameter(a)
cspace.add_hyperparameter(b)

print(cspace["a"] < cspace["b"])
