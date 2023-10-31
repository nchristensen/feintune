import pstats
from pstats import SortKey
p = pstats.Stats("output.prof")
p.strip_dirs().sort_stats(-1).print_stats()
print("CUMULATIVE TIME")
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)

print("TIME")
p.sort_stats(SortKey.TIME).print_stats(30)
