from __future__ import annotations
from typing import Optional, List
import numpy as np

from helpers.Env import *
from helpers.Population import Population, Individual, PopulationIterator


def BB(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None
) -> SearchResult[Particle]:
    """
    Barebones Particle Swarms is a variation of PSO proposed by Kennedy [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1202251
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    particles: List[Particle] = [Particle(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions) for _ in range(populationSize)]
    population: Population = Population(
        individuals=particles,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
    return search(env, population)


def search(env: Env, population: Population) -> SearchResult:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Particle] = population.iterator()
            while (iter.hasNext()):
                particle, neighbors = iter.next()
                particle.update(neighbors.best())
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Particle](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )


class Particle(Individual):
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int
    ):
        super().__init__(
            fitnessFunction = fitnessFunction,
            bounds = bounds,
            numDimensions = numDimensions
        )

    def update(self, best: Particle):
        mu = np.add(self.best, best.best) / 2
        sd = np.abs(np.subtract(self.best, best.best))
        self.pos = np.random.normal(mu,sd)
        fit = self.fitnessFunction(self.pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.best = self.pos
