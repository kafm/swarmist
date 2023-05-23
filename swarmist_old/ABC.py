from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult, MaxEvaluationReached, MinFitnessReached
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def ABC(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    limit: Optional[int] = None,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
) -> SearchResult[FoodSource]:
    """
    Artificial Bee Colony proposed by Karaboga[1]. Implemented as suggested in [1,2].

    [1] https://abc.erciyes.edu.tr/pub/tr06_2005.pdf
    [2] https://link.springer.com/article/10.1007/s10898-007-9149-x
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    numFoodSources:int = int(populationSize/2)
    limit: int = numFoodSources*numDimensions if not limit else limit
    particles: List[FoodSource] = [FoodSource(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i) for i in range(populationSize)]
    population: Population = Population(
        individuals=particles,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
    return search(env, population, limit)


def search(env: Env, population: Population, limit: int) -> SearchResult[FoodSource]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            employedBeesSearch(population)
            onlookerBeesSearch(population)
            scoutBeesSearch(population, limit)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[FoodSource](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def employedBeesSearch(population: Population): 
    iter: PopulationIterator[FoodSource] = population.iterator()
    while (iter.hasNext()):
        foodSource, neighbors = cast(Tuple[FoodSource, Neighborhood], iter.next())
        updateFoodSource(foodSource, neighbors)

def onlookerBeesSearch(population: Population):
    iter: PopulationIterator[FoodSource] = population.filterByRoulette()
    while (iter.hasNext()):
        foodSource, neighbors = cast(Tuple[FoodSource, Neighborhood], iter.next())
        updateFoodSource(foodSource, neighbors)

def scoutBeesSearch(population: Population, limit: int):
    candidate: FoodSource = max(population.individuals, key = lambda i: i.trials)
    if candidate.trials >= limit:
        candidate.reset()
    
def updateFoodSource(foodSource: FoodSource, neighbors: Neighborhood):
    a = neighbors.getRandomIndividual([foodSource.index])
    j = np.random.randint(low=0, high=foodSource.ndims)
    pos = np.copy(foodSource.pos)
    pos[j] += (pos[j] - a.pos[j]) * np.random.uniform(low=-1,high=1)
    pos = np.clip(pos, foodSource.bounds.min, foodSource.bounds.max)
    fit = foodSource.fitnessFunction(pos)
    if(fit < foodSource.fitness): 
        foodSource.fitness = fit
        foodSource.pos = pos
        foodSource.trials = 0
    else: 
        foodSource.trials += 1

class FoodSource(Individual):
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
        self.index:int = index
        self.trials:int = 0

    def reset(self):
        print(f"restartign ABC agent")
        self.__init__(self.fitnessFunction,self.bounds,self.ndims, self.index)
    

