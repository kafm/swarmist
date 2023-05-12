from __future__ import annotations
from typing import Optional, List, Tuple, cast, Callable
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def JAYA(
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
    Jaya implementation proposed by Rao [1].

    [1] http://growingscience.com/beta/ijiec/2072-jaya-a-simple-and-new-optimization-algorithm-for-solving-constrained-and-unconstrained-optimization-problems.html
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    particles: List[Individual] = [Individual(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions) for _ in range(populationSize)]
    population: Population = Population(
        individuals=particles,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population, lambda p, n: jayaUpdate(p, n.best(), n.worse()))


def search(env: Env, population: Population, updateMethod: Callable[[Individual, Neighborhood]]) -> SearchResult[Individual]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Individual] = population.iterator()
            while (iter.hasNext()):
                particle, neighbors = cast(Tuple[Individual, Neighborhood], iter.next())
                updateMethod(particle, neighbors)
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

def jayaUpdate(i: Individual, best: Individual, worse: Individual):
    r1 = np.random.rand(i.ndims)
    r2 = np.random.rand(i.ndims)
    absPos = np.abs(i.pos)
    pos = np.clip(i.pos + r1*(best.pos - absPos) - r2*(worse.pos - absPos), i.bounds.min, i.bounds.max)
    fit = i.fitnessFunction(pos)
    if (fit < i.fitness):
        i.fitness = fit
        i.best = pos
        i.pos = pos

