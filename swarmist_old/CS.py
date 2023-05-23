from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np
import math

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual

def CS(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    alpha: float = 1,
    mu: float = 1.5, #replaced lambda with mu due to reserved python lambda 
    pa: float = .25,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
) -> SearchResult[Cuckoo]:
    """
    Cuckoo Search  proposed by Yang and Deb [1]. Implemented as suggested in [2].

    [1] https://ieeexplore.ieee.org/abstract/document/5393690
    [2] https://link.springer.com/chapter/10.1007/978-3-319-02141-6_1
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    particles: List[Cuckoo] = [Cuckoo(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i) for i in range(populationSize)]
    population: Population = Population(
        individuals=particles,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
    return search(env, population, alpha, mu, pa)


def search(env: Env, population: Population,  alpha: float, mu: float, pa: float) -> SearchResult[Cuckoo]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            cuckooGlobalSearch(population, alpha, mu)
            cuckooLocalSearch(population, pa)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Cuckoo](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def cuckooGlobalSearch(population: Population,  alpha: float, mu: float):
    iter: PopulationIterator[Cuckoo] = population.iterator()
    while (iter.hasNext()):
        cuckoo, neighbors = cast(Tuple[Cuckoo, Neighborhood], iter.next())
        pos = np.clip(
            cuckoo.pos + alpha * levyFlight(mu, cuckoo.ndims) * np.subtract(cuckoo.pos, neighbors.best().pos), 
            cuckoo.bounds.min, 
            cuckoo.bounds.max
        )
        fit = cuckoo.fitnessFunction(pos)
        nest = neighbors.getRandomIndividual(excludeIndexes=[cuckoo.index])
        if fit < nest.fitness:
            nest.fitness = fit
            nest.pos = pos
    
def cuckooLocalSearch(population: Population,  pa: float):
    iter: PopulationIterator[Cuckoo] = population.filterByPropability(pa)
    while (iter.hasNext()):
        cuckoo, neighbors = cast(Tuple[Cuckoo, Neighborhood], iter.next())
        a, b = neighbors.getRandomIndividuals(k=2, excludeIndexes=[cuckoo.index])
        cuckoo.pos = np.clip(
            cuckoo.pos + ( np.random.rand(cuckoo.ndims) * np.subtract(a.pos, b.pos)), 
            cuckoo.bounds.min, 
            cuckoo.bounds.max
        )
        cuckoo.fitness = cuckoo.fitnessFunction(cuckoo.pos)
       
def levyFlight(mean, ndims):
    beta = mean if mean > 1 else 1
    gamma1 = math.gamma(1+beta)
    gamma2 = math.gamma((1+beta)/2)
    sigma = (gamma1*math.sin(math.pi*beta/2)/(gamma2*beta*2**((beta-1)/2))) ** (1/beta)
    u = np.random.normal(0,1, size=ndims) * sigma
    v = np.random.normal(0,1, size=ndims) 
    return u / abs(v) ** (1 / beta)
    
class Cuckoo(Individual):
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int,
        index: int
    ):
        super().__init__(
            fitnessFunction = fitnessFunction,
            bounds = bounds,
            numDimensions = numDimensions
        )
        self.index = index
    

