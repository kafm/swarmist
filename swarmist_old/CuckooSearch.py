from __future__ import annotations
from numba import njit, float64
import math
import numpy as np
from typing import Optional, List
from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached

   
class CuckooSearch:
    def __init__(self, 
        fitnessFunction:FitnessFunction, 
        bounds: Bounds,
        numDimensions: int = 2,
        populationSize:int=30,
        maxGenerations:Optional[int]=None,
        maxEvaluations:Optional[int]=None,
        options:dict={}
    ):
        self.populationSize =  populationSize
        self.numGenerations = maxGenerations
        self.maxEvaluations = maxEvaluations
        self.bounds = bounds
        self.ndims = numDimensions
        self.options = options
        self.fitnessFunction = fitnessFunction
        discoveryProbability = self.options["discoveryProbability"] if "discoveryProbability" in self.options else None
        self.discoveryProbability:float = np.clip(discoveryProbability if discoveryProbability else 0.25, 0, 1)

    def search(self):
        nests:list[Nest] = [Nest(self.fitnessFunction, self.bounds, self.ndims, i, self.options) for i in range(self.populationSize)]
        generations:List[float] = []
        best:Nest = min(nests, key = lambda p : p.fitness)                
        try:
            for _ in range(self.numGenerations):
                for i in range(self.populationSize):
                    cuckoo = nests[i]
                    pos, fit =  cuckoo.getEgg(best)
                    nest: Nest = nests[randIndex(self.populationSize, cuckoo.index)]
                    nest.setEggIfApplies(pos, fit)
                for i in range(self.populationSize):
                    if np.random.uniform() < self.discoveryProbability:
                        nests[i].abandon(nests[randIndex(self.populationSize, cuckoo.index)], nests[randIndex(self.populationSize, cuckoo.index)])
                best = min(nests, key = lambda p : p.fitness)                
        except MaxEvaluationReached:
            best = min(nests, key = lambda p : p.fitness)                   
        return SearchResult[Nest](
            best=best,
            population=nests,
            fitnessByGeneration=generations
        )


class Nest:
    def __init__(self, fitnessFunction:FitnessFunction, bounds: Bounds, ndims: int, index: int, options:dict):
        self.fitnessFunction = fitnessFunction
        self.bounds = bounds
        self.minval:float = bounds.min
        self.maxval:float = bounds.max
        self.ndims:int = ndims
        self.scalingFactor:float = options["scalingFactor"] if "scalingFactor" in options else 1
        self.levyScalingFactor:float = options["levyScalingFactor"] if "levyScalingFactor" in options else 1.5
        self.pa:float = options["discoveryProbability"] if "discoveryProbability" in options else 0.25
        self.pos = getRandomPos(self.minval, self.maxval, self.ndims)
        self.fitness = fitnessFunction(self.pos)
        self.index = index
       
    def setEggIfApplies(self, pos:np.ndarray, fit: float):
        if fit < self.fitness:
            self.pos = pos
            self.fitness = fit

    def getEgg(self, best: Nest) -> tuple(np.ndarray, float): 
        pos = getNextPos(self.pos, best.pos, self.minval, self.maxval, self.scalingFactor, self.levyScalingFactor, self.ndims) 
        fit = self.fitnessFunction(pos)
        return (pos, fit)
     
    def abandon(self, nestJ: Nest, nestK: Nest):
        self.pos = getRandomNextPos(self.pos, nestJ.pos, nestK.pos, self.minval, self.maxval, self.ndims)
        self.fitness = self.fitnessFunction(self.pos)

def getRandomPos(minval:float , maxval:float , ndims: int) -> np.ndarray:
    return np.random.uniform(low=minval, high = maxval, size = ndims)

#@njit(nogil=True)
def getNextPos(currPos: np.ndarray, bestPos: np.ndarray, minval: float, maxval: float, scalingFactor: float, levyScalingFactor: float, ndims: int) -> np.ndarray:
    pos = currPos + scalingFactor * levyFlight(levyScalingFactor, ndims) * np.subtract(currPos, bestPos) 
    return np.clip(pos, minval, maxval) 

@njit(nogil=True)
def getRandomNextPos(currPos: np.ndarray, jPos: np.ndarray, kPos: np.ndarray, minval: float, maxval: float, ndims: int) -> np.ndarray:
    pos = currPos + ( np.random.rand(ndims) * np.subtract(jPos, kPos))
    return np.clip(pos, minval, maxval) 

@njit(nogil=True)
def randIndex(length, excludeIndex = None):
    index = np.random.randint(low=0, high=length)  
    if index == excludeIndex:
        return randIndex(length, excludeIndex)
    return index

#@njit(nogil=True)
def levyFlight(mean, ndims):
    beta = mean if mean > 1 else 1
    gamma1 = math.gamma(1+beta)
    gamma2 = math.gamma((1+beta)/2)
    sigma = (gamma1*math.sin(math.pi*beta/2)/(gamma2*beta*2**((beta-1)/2))) ** (1/beta)
    u = np.random.normal(0,1, size=ndims) * sigma
    v = np.random.normal(0,1, size=ndims) 
    return u / abs(v) ** (1 / beta)
    
    