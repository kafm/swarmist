from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual

#a=2
#b=.5
def WO(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    a: float = 2,
    spiral: float = .5,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
) -> SearchResult[Whale]:
    """
    Whale Optimization Algorithm implementation proposed by Mirjalili and Lewis [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0965997816300163
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    individuals: List[Whale] = [Whale(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i) for i in range(populationSize)]
    population: Population = Population(
        individuals=individuals,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population, a, spiral)


def search(env: Env, population: Population, a: float, spiral: float) -> SearchResult[Whale]:
    try:
        fitnessByGeneration: List[float] = []
        a_step = a/env.maxGenerations
        while (env.next()):
            iter: PopulationIterator[Whale] = population.iterator()
            a -= a_step #TODO - env.currGen * (2 / env.maxGenerations)
            while (iter.hasNext()):
                individual, neighbors = cast(Tuple[Whale, Neighborhood], iter.next())
                whaleUpdate(individual, neighbors, a, spiral)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Whale](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def whaleUpdate(w: Whale, neighbors: Neighborhood, a: float, spiral: float):
    p = np.random.uniform()
    if p < .5:
        r = np.random.uniform(size=w.ndims)
        A = ( 2 * np.multiply(a, r) ) - a
        if np.linalg.norm(A) < 1:
            w.pos = encircle(w, neighbors.best(), A)
        else:
            w.pos = explore(w, neighbors, A)
    else:
         w.pos = attack(w, neighbors.best(), spiral)
    w.fitness = w.fitnessFunction(w.pos)
 
def encircle(w: Whale, best: Whale, A: List[float])->List[float]:
    C =  2 * np.random.random(size=w.ndims)
    D = np.abs( C * best.pos - w.pos )
    pos = best.pos - A * D
    return np.clip(pos, w.bounds.min, w.bounds.max)

def attack(w: Whale, best: Whale, spiral: float)->List[float]:
    D =np.abs(best.pos - w.pos)
    l = np.random.uniform(-1.0, 1.0, size=w.ndims)
    pos = D * np.exp(spiral*l) * np.cos(2.0*np.pi*l) + best.pos
    return np.clip(pos, w.bounds.min, w.bounds.max)

def explore(w: Whale, neighbors: Neighborhood, A: List[float]):
    rW = neighbors.getRandomIndividual(excludeIndexes=[w.index])
    C = 2 * np.random.random(size=w.ndims)
    D = np.abs( C * rW.pos - w.pos )   
    pos = rW.pos - A * D
    return np.clip(pos, w.bounds.min, w.bounds.max)


class Whale(Individual):
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