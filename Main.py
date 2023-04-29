from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from swarmist_bck import SearchResult, PSO, benchmark

algo = PSO
numDimensions = 20
populationSize = 40
numGenerations = 1000
maxEvaluations = None #50000
minFitness = None



func, bounds = benchmark.sphere()
res: SearchResult = algo(
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
