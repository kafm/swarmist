from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def SCA(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
) -> SearchResult[Individual]:
    """
    Sine Cosine algorithm implementation proposed by Mirjalili [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0950705115005043
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    agents: List[Individual] = [Individual(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions) for _ in range(populationSize)]
    population: Population = Population(
        individuals=agents,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population)


def search(env: Env, population: Population, a: int=2) -> SearchResult[Individual]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Individual] = population.iterator()
            r1 = a - env.currGen * ((a) / env.maxGenerations) 
            while (iter.hasNext()):
                individual, neighbors = cast(Tuple[Individual, Neighborhood], iter.next())
                scUpdate(individual, neighbors.best(), r1)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Individual](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def scUpdate(a: Individual, best: Individual, r1: float):
    r2 = np.random.uniform(low=0, high=2*np.pi, size=a.ndims)
    r3 = np.random.uniform(low=-2, high=2, size=a.ndims)
    r4 = np.random.uniform(size=a.ndims)
    pos = np.copy(a.pos)
    for i in range(a.ndims): 
        sc = np.sin(r2[i]) if r4[i] < .5 else np.cos(r2[i])
        pos[i] += ( 
            r1 * 
            sc * 
            abs(r3[i] * best.pos[i] - pos[i]) 
        )
    pos = np.clip(pos, a.bounds.min, a.bounds.max)
    fit = a.fitnessFunction(pos)
    if fit < a.fitness:
        a.pos = pos
        a.fitness = fit
 
