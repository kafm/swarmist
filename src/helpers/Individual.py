from __future__ import annotations
from typing import List
from .Env import FitnessFunction, Bounds
import numpy as np

class Individual: 
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int
    ):
        self.bounds = bounds
        self.ndims = numDimensions
        self.bounds: Bounds = bounds
        self.fitnessFunction = fitnessFunction
        self.pos = np.random.uniform(low=bounds.min, high=bounds.max, size=numDimensions)
        self.best = self.pos
        self.fitnessFunction = fitnessFunction
        self.fitness = fitnessFunction(self.pos)
       


class Neighborhood: 
    def __init__(self, index: int):
        self.individuals: List[Individual] = []
        self.rank: List[Individual] = []
        self.index:int = index

    def append(self, individual: Individual):
        self.individuals.append(individual)

    def resolveRank(self):
       #self. np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
       self.nbest = min(self.individuals, key = lambda i : i.fitness)
       self.nworse = max(self.individuals, key = lambda i : i.fitness)

    def getRandomIndividuals(self, k: int=1, excludeIndexes: List[int] = None, replace:bool=False)->List[Individual]:
        inds:List[Individual] = self.individuals if not excludeIndexes else [
            self.individuals[i] for i in range(len(self.individuals)) if i not in excludeIndexes
        ]
        return np.random.choice(inds, k, replace=replace)

    def getRandomIndividual(self, excludeIndexes: List[int] = None)->Individual:
        r = np.random.randint(low=0, high=len(self.individuals))
        if not excludeIndexes or r not in excludeIndexes:
            return self.individuals[r]
        return self.getRandomIndividual(excludeIndexes=excludeIndexes)

    def best(self)->Individual:
        return self.nbest

    def worse(self)->Individual:
        return self.nworse
    
    def meanPos(self)->List[float]:
        n = len(self.individuals)
        s = sum([i.pos for i in self.individuals])
        return np.divide(s,n)
        