from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual


def TLBO(
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
    Teaching–learning-based optimization implementation proposed by Rao, Savsani and Vakharia [1].

    [1] https://www.sciencedirect.com/science/article/pii/S0010448510002484
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    Students: List[Student] = [Student(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i) for i in range(populationSize)]
    population: Population = Population(
        individuals=Students,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
  
    return search(env, population)


def search(env: Env, population: Population) -> SearchResult[Student]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            teacherPhase(population)
            studentPhase(population)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Student](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def teacherPhase(population: Population):
    iter: PopulationIterator[Student] = population.iterator()
    while (iter.hasNext()):
        individual, neighbors = cast(Tuple[Student, Neighborhood], iter.next())
        tf = np.random.choice([1,2])
        mu = neighbors.meanPos()
        r = np.random.uniform(size=individual.ndims)
        pos = np.clip(
            individual.pos + ( r * (neighbors.best().pos - tf*mu) ),
            individual.bounds.min, individual.bounds.max
        )
        updatePosIfApplies(individual, pos)

def studentPhase(population: Population):
    iter: PopulationIterator[Student] = population.iterator()
    while (iter.hasNext()):
        individual, neighbors = cast(Tuple[Student, Neighborhood], iter.next())
        partner = neighbors.getRandomIndividual(excludeIndexes=[individual.index])
        diff = individual.pos-partner.pos if individual.fitness < partner.fitness else partner.pos-individual.pos
        r = np.random.uniform(size=individual.ndims)
        pos = np.clip(
            individual.pos + r * diff,
            individual.bounds.min, individual.bounds.max
        )
        updatePosIfApplies(individual, pos)


def updatePosIfApplies(i: Student, pos: List[float])->bool:
    fit = i.fitnessFunction(pos)
    if fit < i.fitness:
        i.pos = pos
        i.fitness = fit
        return True
    return False

class Student(Individual):
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