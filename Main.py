from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from swarmist.core import *
from swarmist.algos.helpers import lbest
from swarmist.algos.pso import Pso, Fips, Barebones
from swarmist.utils.benchmark import sphere, ackley, schwefel
from swarmist_bck.JAYA import SearchResult, JAYA

numDimensions = 20
populationSize = 40
numGenerations = 1000
maxEvaluations = 50000
minFitness = None
func, bounds = sphere()

res_original: SearchResult = JAYA(
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
        size(populationSize),
        init(lambda ctx: np.random.uniform(ctx.bounds.min, ctx.bounds.max, size=ctx.ndims)),
        None, #topology(lbest()),
        *Barebones().pipeline()
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
    print(f"New={results[-1].fit}, Old={orig_results.fitnessByGeneration[-1]}")

def raise_error(e): 
    raise e

res_new.either(
    raise_error, 
    lambda res: print_best(res,res_original)
)



