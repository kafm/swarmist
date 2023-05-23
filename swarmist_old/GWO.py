from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np
import copy

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def GWO(
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
    Grey Wolf Optimizer implementation proposed by Mirjalili, Mohammad Mirjalili and Lewis [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0965997813001853
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    individuals: List[Individual] = [Individual(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions) for i in range(populationSize)]
    population: Population = Population(
        individuals=individuals,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population)


def search(env: Env, population: Population) -> SearchResult[Individual]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Individual] = population.iterator()
            a = 2 - env.currGen * (2 / env.maxGenerations) 
            while (iter.hasNext()):
                individual, neighbors = cast(Tuple[Individual, Neighborhood], iter.next())
                gwoUpdate(individual, neighbors, a)
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

def gwoUpdate(w: Individual, neighbors: Neighborhood, a: float=2):
    A1, A2, A3 = 2 * a * np.random.random() - a, 2 * a * np.random.random() - a, 2 * a * np.random.random() - a
    C1, C2, C3 = 2 * np.random.random(), 2 * np.random.random(), 2 * np.random.random()
    X1, X2, X3 = np.zeros(w.ndims), np.zeros(w.ndims), np.zeros(w.ndims)
    alpha, beta, gamma = copy.copy(neighbors.rank[:3])
    pos = np.zeros(w.ndims)
    for j in range(w.ndims):
        X1[j] = alpha.pos[j] - A1 * abs(C1 * alpha.pos[j] - w.pos[j])
        X2[j] = beta.pos[j] - A2 * abs(C2 *  beta.pos[j] - w.pos[j])
        X3[j] = gamma.pos[j] - A3 * abs(C3 * gamma.pos[j] - w.pos[j])
        pos[j] = (X1[j] + X2[j] + X3[j])/3
    pos = np.clip(pos, w.bounds.min, w.bounds.max)
    fit = w.fitnessFunction(pos)
    if fit < w.fitness:
        w.pos = pos
        w.fitness = fit