from autotune import TuningProblem
from autotune.space import *
from skopt.space import Real
from ytopt.search.ambs import AMBS
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

input_space = cs.ConfigurationSpace("autotuning_space")
input_space.add_hyperparameter(csh.OrdinalHyperparameter("a", [0]))
output_space = Space([Real(0.0, inf, name="avg_time")])


def obj_func(p):
    return 1


problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=obj_func,
    constraints=None,
    model=None)

searcher = AMBS(problem=problem, evaluator="ray")
# exit()
searcher.main()

"""
def offline_tuning(queue, knl, max_flop_rate=np.inf, device_memory_bandwidth=np.inf,
                     device_latency=0, timeout=np.inf):

    input_space = createConfigSpace(queue, knl)
    print(input_space)
    for entry in input_space.values():
        print(type(entry), entry)
    #exit()
    
    output_space = Space([Real(0.0, inf, name="avg_time")])


    def obj_func(p):
        params = (p["batch_size"],
                  p["kio"]*p["kii"],
                  p["kii"],
                  p["iio"]*p["iii"],
                  p["iii"],
                  p["ji"],)
        tlist = get_trans_list(knl, params)
        results = run_single_param_set_v2(queue, knl, tlist, max_flop_rate=max_flop_rate,
                    device_memory_bandwidth=device_memory_bandwidth, device_latency=device_latency, timeout=timeout)

        return results["data"]["avg_time"]


    at_problem = TuningProblem(
        task_space=None,
        input_space=input_space,
        output_space=output_space,
        objective=obj_func,
        constraints=None,
        model=None)

    # Not quite certain what the difference is between
    # these but AsyncSearch seems to support MPI. Can't find NeuralNetworksDropoutRegressor
    #searcher = AsyncSearch(at_problem, "subprocess")
    searcher = AMBS(problem=at_problem, evaluator="subprocess")
    searcher.main()
"""
