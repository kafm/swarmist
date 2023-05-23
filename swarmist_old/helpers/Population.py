from __future__ import annotations
from typing import Callable, List, Tuple, Optional, cast
from .Individual import Individual, Neighborhood
from .Topology import buildTopology
import numpy as np
import sys

class Population: 
    def __init__(self,
        individuals: List[Individual],
        topology: Optional[str] = None,
        neighborhoodRange: Optional[int] = None
    ):
        self.individuals: List[Individual] = individuals
        self.size: int = len(individuals) 
        self.topologyName:str = "lbest" if topology == "lbest" else "gbest"
        self.neighborhood:List[Neighborhood] = buildTopology(individuals, self.topologyName, neighborhoodRange)
        self.rank:List[Individual]= []

    def getAll(self)->List[Individual]:
        return self.individuals

    def resolveRank(self)->List[Individual]:
        best: Individual = None
        worse: Individual = None
        minFit:float = sys.float_info.max
        maxFit:float = 0
        previousN: Neighborhood = None
        for n in self.neighborhood: 
            if previousN == None or previousN.index != n.index:
                n.resolveRank()
            nBest = n.best()
            nWorse = n.worse()
            if minFit > nBest.fitness:
                best = nBest
            if maxFit < nWorse.fitness:
                worse = nWorse
        self.rank = [ best, worse ]
        return self.rank

    def best(self)->Individual:
        self.resolveRank()
        return self.rank[0]


    def worse(self)->Individual:
        self.resolveRank()
        return self.rank[-1]

    def filterByPropability(self, p: float)->PopulationIterator:
        return ProbabilityFilterInterator(self.individuals, self.neighborhood, p)

    def filterByRoulette(self)->PopulationIterator:
        r = self.resolveRank()
        return RouletteFilterInterator(self.individuals, self.neighborhood, r[0].fitness, r[-1].fitness)

    def iterator(self)->PopulationIterator:
        return PopulationIterator(self.individuals, self.neighborhood)

class PopulationIterator: 
    def __init__(self,
        individuals: List[Individual],
        neighborhood: List[Neighborhood],
    ):
        self.individuals: List[Individual] = individuals
        self.neighborhood: List[Neighborhood] = neighborhood
        self.length = len(individuals)
        self.index:int = 0

    def hasNext(self)->bool:
        return self.index < self.length

    def next(self)->Tuple[Individual, Neighborhood]:
        i = self.index 
        self.index += 1
        return (self.individuals[i], self.neighborhood[i])

class ProbabilityFilterInterator(PopulationIterator): 
    def __init__(self,
        individuals: List[Individual],
        neighborhood: List[Neighborhood],
        p: float
    ):
        super().__init__(individuals, neighborhood)
        self.p = p

    def hasNext(self)->bool:
        if self.index < self.length:
            if np.random.uniform() < self.p:
                return True
            elif self.index < self.length:
                self.index += 1
                return self.hasNext()
        return False

class RouletteFilterInterator(PopulationIterator): 
    def __init__(self,
        individuals: List[Individual],
        neighborhood: List[Neighborhood],
        minFit: float, 
        maxFit: float
    ):
        super().__init__(individuals, neighborhood)
        fits = [(maxFit - i.fitness) / (maxFit - minFit) for i in self.individuals]
        total = sum(fits)
        self.probs = [fit/total for fit in fits]
        self.indexes = np.arange(0, self.length)

    def next(self)->Tuple[Individual, Neighborhood]:
        i = np.random.choice(self.indexes, p=self.probs, replace=True)
        self.index += 1
        return (self.individuals[i], self.neighborhood[i]) 


