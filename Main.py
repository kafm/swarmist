from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from swarmist.core import *
from swarmist.algos.pso import Pso, Barebones, Fips
from swarmist.utils.benchmark import sphere

from swarmist_bck.PSO import SearchResult, PSO

numDimensions = 20
populationSize = 40
numGenerations = 1001
maxEvaluations = 50000
minFitness = None
func, bounds = sphere()

res_original: SearchResult = PSO(
    fitnessFunction=func,
    bounds = bounds,
    numDimensions = numDimensions,
    populationSize = populationSize,
    #maxGenerations = numGenerations,
    maxEvaluations = maxEvaluations
)

res_new = search(
    search_space=space(
        dimensions=dimensions(numDimensions),
        bounds=bounded(bounds.min, bounds.max),
        #constraints=constrained_by(lambda pos: pos ),
        fit_function=minimize(func)
    ),
    stop_condition=until(
        #max_gen=numGenerations,
        max_evals=maxEvaluations
    ),
    search_strategy=using(
        size(40),
        init(lambda ctx: np.random.uniform(ctx.bounds.min, ctx.bounds.max, size=ctx.ndims)),
        None,
        update(
            select(all()),
            apply(Pso().update),
            #apply(lambda ctx: ctx.agent),
            #where(lambda a: a.improved)
        )
    ), 
    #replace=False
)


def plot_algos(results: SearchResults, orig_results: SearchResult):
    np.seterr('raise')
    print(results)
    plt.figure("Sphere")
    #plt.plot([r.best.fit for r in res.results])
    plt.plot([r.fit for r in results], label="New")
    plt.plot(orig_results.fitnessByGeneration, label="Original")
    plt.legend(loc="upper left")
    plt.show()

def print_best(results: SearchResults, orig_results: SearchResult):
    print(f"New={results[-1].fit}, Old={orig_results.fitnessByGeneration[-1]}")

def raise_error(e): 
    raise e

res_new.either(
    raise_error, 
    lambda res: print_best(res,res_original)
)



