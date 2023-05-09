from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from swarmist.core import *
from swarmist.algos.pso import Pso, Barebones, Fips
from swarmist.utils.benchmark import sphere

numDimensions = 20
populationSize = 40
numGenerations = 1001
maxEvaluations = None #50000
minFitness = None
func, bounds = sphere()

#TODO compare the 2 implementation.... Canonical and library... 2 graphs

res = search(
    search_space=space(
        dimensions=dimensions(numDimensions),
        bounds=bounded(bounds.min, bounds.max),
        #constraints=constrained_by(lambda pos: pos ),
        fit_function=minimize(func)
    ),
    stop_condition=until(
        max_gen=numGenerations,
        #max_evals=maxEvaluations
    ),
    search_strategy=using(
        size(40),
        init(lambda ctx: np.random.uniform(ctx.bounds.min, ctx.bounds.max, size=ctx.ndims)),
        None,
        update(
            select(all()),
            apply(Pso().update),
            #apply(lambda ctx: ctx.agent),
            where(lambda a: a.improved)
        )
    ), 
    replace=False
)

if isinstance(res, SearchFailed): 
    raise res.error

print(res.results)
plt.figure("Sphere")
#plt.plot([r.best.fit for r in res.results])
plt.plot([r for r in res.results])
plt.show()
