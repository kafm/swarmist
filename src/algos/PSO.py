from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np

from helpers.Env import *
from helpers.Population import Population, Individual, PopulationIterator


def PSO(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
    c1: float = 2.05,
    c2: float = 2.05,
    k: float = .729
) -> SearchResult[Particle]:
    """
    Particle Swarm Optimization implementation based on the constriction coefficient proposal made by Clerc and Kennedy [1].
    In this implementation we use the Type 1''  constriction coefficient.

    [1] https://ieeexplore.ieee.org/abstract/document/985692
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    phi:float = c1 + c2
    chi:float = np.sqrt(k) if phi <= 4 else np.sqrt(
            (2*k) / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))
    particles: List[Particle] = [Particle(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        c1=c1,
        c2=c2,
        chi=chi
    ) for _ in range(populationSize)]
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
        numDimensions: int,
        c1: float,
        c2: float,
        chi: float
    ):
        super().__init__(
            fitnessFunction = fitnessFunction,
            bounds = bounds,
            numDimensions = numDimensions
        )
        self.v = np.zeros(numDimensions)
        self.c1 = c1
        self.c2 = c2
        self.chi = chi

    def update(self, best: Particle):
        self.v = self.chi * (
            self.v +
            self.c1 * np.random.rand(self.ndims) * (self.best - self.pos) +
            self.c2 * np.random.rand(self.ndims) * (best.best - self.pos)
        )
        self.pos = np.clip(self.pos + self.v, self.bounds.min, self.bounds.max)
        fit = self.fitnessFunction(self.pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.best = self.pos