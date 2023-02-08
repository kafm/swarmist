from __future__ import annotations
import matplotlib.pyplot as plt

#from algos.PSO import PSO
#from algos.BB import BB
from algos.FIPS import FIPS
from helpers.Env import Bounds, SearchResult

numDimensions = 20
populationSize = 40
numGenerations = 1000
maxEvaluations = None #50000
minFitness = None

def sphere(x):
 return sum(x ** 2)

res: SearchResult = FIPS(
    fitnessFunction=sphere,
    bounds = Bounds(min=-5.12, max=5.12),
    topology="lbest",
    neighborhoodRange=2,
    numDimensions = numDimensions,
    populationSize = populationSize,
    maxGenerations = numGenerations,
    maxEvaluations = maxEvaluations,
    minFitness = minFitness
)

print("PSO->Sphere: ", res.best.fitness)
plt.figure("Sphere")
plt.plot(res.fitnessByGeneration)
plt.show()