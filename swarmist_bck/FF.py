from __future__ import annotations
from typing import Optional, List, Tuple, cast, Callable
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def FF(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    alpha: float = 1,
    beta: float = 1,
    gamma:float = .01,
    delta: float = .97,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
) -> SearchResult[Firefly]:
    """
    Firefly implementation proposed by Yang [1]. Delta parameter included as suggested in [2].

    [1] https://link.springer.com/chapter/10.1007/978-3-642-04944-6_14
    [2] https://link.springer.com/chapter/10.1007/978-3-319-02141-6_1
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    particles: List[Firefly] = [Firefly(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i,
        alpha=alpha) for i in range(populationSize)]
    population: Population = Population(
        individuals=particles,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population, lambda ff, n: fireflyUpdate(ff, n, beta, gamma, delta))


def search(env: Env, population: Population, updateMethod: Callable[[Firefly, Neighborhood]]) -> SearchResult[Firefly]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Firefly] = population.iterator()
            while (iter.hasNext()):
                particle, neighbors = cast(Tuple[Firefly, Neighborhood], iter.next())
                updateMethod(particle, neighbors)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Firefly](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def fireflyUpdate(ff: Firefly, neighbors: Neighborhood, beta: float, gamma: float, delta: float):
    ff.alpha *= delta
    pos = np.copy(ff.pos)
    for n in neighbors.individuals:
        if n.fitness < ff.fitness:
            d = n.pos - pos
            ffBeta = beta * np.exp(-gamma *  np.square(d))
            e = ff.alpha * (np.random.random(ff.ndims) - 0.5)
            pos = pos + ffBeta * d + e
        oldFit = ff.fitness
        ff.fitness = ff.fitnessFunction(pos)
        ff.pos = pos
        if oldFit > ff.fitness:
            ff.best = pos
    
class Firefly(Individual):
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int,
        index: int,
        alpha: float
    ):
        super().__init__(
            fitnessFunction = fitnessFunction,
            bounds = bounds,
            numDimensions = numDimensions
        )
        self.index = index
        self.alpha = alpha
    