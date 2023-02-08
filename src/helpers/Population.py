from __future__ import annotations
from abc import ABC
from typing import TypeVar, Generic, Callable, List, Tuple, Optional
from .Env import FitnessFunction, Bounds
import sys
import numpy as np

class Population: 
    def __init__(self,
        individuals: List[Individual],
        topology: Optional[str] = None,
        neighborhoodRange: Optional[int] = None
    ):
        self.individuals: List[Individual] = individuals
        self.size: int = len(individuals) 
        self.topologyName:str = "lbest" if topology == "lbest" else "gbest"
        self.neighborhood:Topology[Individual] = buildTopology(individuals, self.topologyName, neighborhoodRange)

    def getAll(self)->List[Individual]:
        return self.individuals

    def best(self)->Individual:
        best: Individual = None
        minFit:float = sys.float_info.max
        previousN: Neighborhood = None
        for n in self.neighborhood: 
            if previousN == None or previousN.index != n.index:
                n.resolveBest()
            nBest = n.best()
            if minFit > nBest.fitness:
                best = nBest
        return best

    def iterator(self)->PopulationIterator:
        return PopulationIterator(self.individuals, self.neighborhood)

class Individual(ABC): 
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int
    ):
        self.bounds = bounds
        self.ndims = numDimensions
        self.pos = np.random.uniform(low=bounds.min, high=bounds.max, size=numDimensions)
        self.bounds: Bounds = bounds
        self.best = self.pos
        self.fitnessFunction = fitnessFunction
        self.fitness = fitnessFunction(self.pos)

class PopulationIterator: 
    def __init__(self,
        individuals: List[Individual],
        neighborhood: Topology[Individual]
    ):
        self.individuals: List[Individual] = individuals
        self.neighborhood: Topology[Individual] = neighborhood
        self.length = len(individuals)
        self.index:int = 0

    def hasNext(self)->bool:
        return self.index < self.length

    def next(self)->Tuple[Individual, Neighborhood]:
        i = self.index 
        self.index += 1
        return (self.individuals[i], self.neighborhood[i])

def  buildTopology(individuals: List[Individual], topologyName:str, neighborhoodRange:Optional[int]=None)->Topology:
    if topologyName == "lbest":
        return LBEST(individuals, 2 if neighborhoodRange == None else neighborhoodRange)
    else: 
        return GBEST(individuals)
        
def GBEST(individuals: List[Individual])->Topology:
    topology = []
    neighborhood: Neighborhood = Neighborhood(0)
    for i in individuals: 
        neighborhood.append(i)
        topology.append(neighborhood)
    return topology

def LBEST(individuals: List[Individual], k:int=2)->Topology:
    topology:List[Neighborhood] = []
    n = len(individuals)
    for i in range(n): 
        neighborhood:Neighborhood = Neighborhood(i)
        for j in range(k):
            neighborhood.append(individuals[(i + j) % n])
            neighborhood.append(individuals[(i - j) % n])
        topology.append(neighborhood)
    return topology

class Neighborhood: 
    def __init__(self, index: int):
        self.individuals: List[Individual] = []
        self.nbest = None
        self.index = index

    def append(self, individual: Individual):
        self.individuals.append(individual)
        if self.nbest == None or self.nbest.fitness > individual.fitness:
            self.nbest = individual

    def resolveBest(self):
       self.nbest = min(self.individuals, key = lambda p : p.fitness)

    def best(self)->Individual:
        return self.nbest

Topology = List[Neighborhood]
