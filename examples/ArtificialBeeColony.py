from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import swarmist_old as sw

algo = sw.ABC
numDimensions = 20
populationSize = 40
numGenerations = 1000
maxEvaluations = None #50000
minFitness = None



func, bounds = sw.benchmark.sphere()
res: sw.SearchResult = algo(
    fitnessFunction=func,
    bounds = bounds,
    numDimensions = numDimensions,
    populationSize = populationSize,
    maxGenerations = numGenerations,
    maxEvaluations = maxEvaluations,
    minFitness = minFitness
)

print("Sphere: ", res.best.fitness)
plt.figure("Sphere")
plt.plot(res.fitnessByGeneration)
plt.show()
