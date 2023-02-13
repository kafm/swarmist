from __future__ import annotations
from typing import List, Tuple, Optional
from .Individual import Individual, Neighborhood
from .Topology import buildTopology
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

class PopulationIterator: 
    def __init__(self,
        individuals: List[Individual],
        neighborhood: List[Neighborhood]
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