from __future__ import annotations
from typing import Optional, List, Tuple
from numba import njit
import numpy as np

from helpers.Env import *
from helpers.Population import Population, Individual, PopulationIterator


def FIPS(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: str = "lbest",
    neighborhoodRange: Optional[int] = None,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
    phi: float = 4.1,
    k: float = .729
) -> SearchResult[Particle]:
    """
    The Fully Informed Particle Swarm is a variation of PSO proposed by Mendes, Kennedy and Neves [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1304843
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    chi:float = np.sqrt(k) if phi <= 4 else np.sqrt(
            (2*k) / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))
    particles: List[Particle] = [Particle(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        phi=phi,
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
                particle.update(neighbors.individuals)
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
        phi: float,
        chi: float
    ):
        super().__init__(
            fitnessFunction = fitnessFunction,
            bounds = bounds,
            numDimensions = numDimensions
        )
        self.v = np.zeros(numDimensions)
        self.phi = phi
        self.chi = chi

    def update(self, neighbors: List[Particle]):
        n = len(neighbors)
        c = self.phi / n
        pm = np.zeros(self.ndims)
        w = np.zeros(self.ndims)
        for p in neighbors:
            wi = c * np.random.rand(self.ndims)
            pm += wi * p.best
            w += wi
        pm /= w
        self.v = self.chi * ( self.v + self.phi * (pm - self.pos) )
        self.pos = np.clip(self.pos + self.v, self.bounds.min, self.bounds.max)
        fit = self.fitnessFunction(self.pos)
        if (fit < self.fitness):
            self.fitness = fit
            self.best = self.pos
