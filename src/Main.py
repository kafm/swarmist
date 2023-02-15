from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from algos.ABC import ABC as algo
#from algos.BB import BB
#from algos.FIPS import FIPS

#from algos.DE import DE

from helpers.Env import Bounds, SearchResult

numDimensions = 20
populationSize = 40
numGenerations = 1000
maxEvaluations = None #50000
minFitness = None

def sphere(x): # (sphere, Bounds(min=-5.12, max=5.12))
 return sum(x ** 2)

def schwefel(x): # (schwefel, Bounds(min=-500, max=500))
    n = len(x)
    return 418.9829 * n - np.sum( x * np.sin( np.sqrt( np.abs( x ))))

def rastrigin(x): # (rastrigin, Bounds(min=-5.12, max=5.12))
    n = len(x)
    return 10 * n + np.sum( x**2 - 10 * np.cos( 2 * np.pi * x ))

def michalewicz(x, m=10): # (michalewicz, Bounds(min=0, max=np.pi))
    ii = np.arange(1,len(x)+1)
    return - sum(np.sin(x) * (np.sin(ii*x**2/np.pi))**(2*m))

func, bounds = (sphere, Bounds(min=-5.12, max=5.12))
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
