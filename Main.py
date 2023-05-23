from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from swarmist_bck.core import *
from swarmist_bck.algos.helpers import *
from swarmist_bck.algos.pso import Pso, Fips, Barebones
from swarmist_bck.algos.jaya import Jaya
from swarmist_bck.algos.abc import Abc
from swarmist_bck.algos.de import De
from swarmist_bck.algos.tlbo import Tlbo
from swarmist_bck.algos.gwo import Gwo
from swarmist_bck.algos.wo import Wo
from swarmist_bck.algos.fa import Fa
from swarmist_bck.algos.sca import Sca
from swarmist_bck.algos.cs import Cs
from swarmist_bck.utils.benchmark import sphere, ackley, schwefel
#from swarmist_bck.PSO import SearchResult, PSO
#from swarmist_bck.TLBO import SearchResult, TLBO
#from swarmist_bck.WO import SearchResult, WO
from swarmist_old.CS import SearchResult, CS
from swarmist_old.CuckooSearch import CuckooSearch

#TODO Cuckoo have behaviours not supported. Its not possible to perform eval within pipeline and its not possible to update another element from within pipeline

numDimensions = 20
populationSize = 40
halfPopulation = 20
numGenerations = 1000
maxEvaluations = 50000
minFitness = None
func, bounds = sphere()

res_old_old: SearchResult = CuckooSearch(
    fitnessFunction=func,
    bounds = bounds,
    numDimensions = numDimensions,
    populationSize = populationSize,
    maxGenerations = numGenerations,
).search()

print("Old old ended", res_old_old.best.fitness)

res_original: SearchResult = CS(
    fitnessFunction=func,
    bounds = bounds,
    numDimensions = numDimensions,
    populationSize = populationSize,
    maxGenerations = numGenerations,
    #maxEvaluations = maxEvaluations
) 

print("Old ended", res_original.best.fitness)

res_new = search(
    search_space=space(
        dimensions=dimensions(numDimensions),
        bounds=bounded(bounds.min, bounds.max),
        #constraints=constrained_by(lambda pos: pos ),
        fit_function=minimize(func)
    ),
    stop_condition=until(
        #max_evals=maxEvaluations,
        max_gen=numGenerations
    ),
    search_strategy=using(
        size(populationSize),
        init(lambda ctx: np.random.uniform(low=ctx.bounds.min, high=ctx.bounds.max, size=ctx.ndims)),
        None, #topology(lbest()),
        *Cs().pipeline()
        # update(
        #     select(all()),
        #     apply(Pso().update)
        # )
    ), 
    #replace=False
)


def plot_algos(results: SearchResults, orig_results: SearchResult):
    np.seterr('raise')
    plt.figure("Sphere")
    #plt.plot([r.best.fit for r in res.results])
    plt.plot([r.fit for r in results], label="New")
    plt.plot(orig_results.fitnessByGeneration, label="Original")
    plt.legend(loc="upper left")
    plt.show()

def print_best(results: SearchResults, orig_results: SearchResult):
    print(f"New size={len(results)}, Old size={len(orig_results.fitnessByGeneration)}")
    print(f"Min new={min(results, key=lambda r: r.fit).fit}")
    print(f"New={results[-1].fit}, Old={orig_results.fitnessByGeneration[-1]}")

def raise_error(e): 
    raise e

res_new.either(
    raise_error, 
    lambda res: print_best(res,res_original)
)