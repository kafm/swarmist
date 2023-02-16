from __future__ import annotations
from typing import Optional, List, Tuple, cast
import numpy as np

from .helpers.Env import Env, Bounds, FitnessFunction, SearchResult
from .helpers.Population import Population, PopulationIterator
from .helpers.Individual import Neighborhood, Individual

def PSO(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    variant: str = "canonical",
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None,
    c1: float = 2.05,
    c2: float = 2.05,
    phi: float = 4.1,
    k: float = .729
) -> SearchResult[Particle]:
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
        topology="lbest" if topology == None and variant == "fips" else topology,
        neighborhoodRange=neighborhoodRange
    )
    updateMethod = getUpdateMethod(variant, c1, c2, phi, k)
    return search(env, population, updateMethod)


def search(env: Env, population: Population, updateMethod: Callable[[Particle, Neighborhood]]) -> SearchResult[Particle]:
    try:
        fitnessByGeneration: List[float] = []
        while (env.next()):
            iter: PopulationIterator[Particle] = population.iterator()
            while (iter.hasNext()):
                particle, neighbors = cast(
                    Tuple[Particle, Neighborhood], iter.next())
                updateMethod(particle, neighbors)
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


def getUpdateMethod(
    variant: str,
    c1: float = 2.05,
    c2: float = 2.05,
    phi: float = 4.1,
    k: float = .729
) -> Callable[[Particle, Neighborhood]]:
    if variant == "fips":
        return lambda p, n: fipsUpdate(p, n.individuals, phi, getChi(phi, k))
    elif variant == "barebones":
        return lambda p, n: barebonesUpdate(p, n.best())
    else:
        return lambda p, n: psoUpdate(p, n.best(), c1, c2, getChi(phi, k))


def psoUpdate(p: Particle, best: Particle, c1: float, c2: float, chi: float):
    """
    Particle Swarm Optimization implementation based on the constriction coefficient proposal made by Clerc and Kennedy [1].
    In this implementation we use the Type 1''  constriction coefficient.

    [1] https://ieeexplore.ieee.org/abstract/document/985692
    """
    persistence = p.pos - p.prevPos
    p.prevPos = p.pos
    p.pos = np.clip(p.pos + chi * (
        persistence +
        c1 * np.random.rand(p.ndims) * (p.best - p.pos) +
        c2 * np.random.rand(p.ndims) * (best.best - p.pos)
    ), p.bounds.min, p.bounds.max)
    fit = p.fitnessFunction(p.pos)
    if (fit < p.fitness):
        p.fitness = fit
        p.best = p.pos


def fipsUpdate(p: Particle, neighbors: Neighborhood, phi: float, chi: float):
    """
    The Fully Informed Particle Swarm is a variation of PSO proposed by Mendes, Kennedy and Neves [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1304843
    """
    n = len(neighbors)
    c = phi / n
    pm = np.zeros(p.ndims)
    w = np.zeros(p.ndims)
    for p in neighbors:
        wi = np.random.rand(p.ndims)
        pm += wi * p.best
        w += wi
    pm /= w
    persistence = p.pos - p.prevPos
    sct = np.random.rand(p.ndims) * c * w * \
        (pm - p.pos)  # Social central tendency
    p.prevPos = p.pos
    p.pos = np.clip(p.pos + chi * (persistence + sct),
                    p.bounds.min, p.bounds.max)
    fit = p.fitnessFunction(p.pos)
    if (fit < p.fitness):
        p.fitness = fit
        p.best = p.pos


def barebonesUpdate(p: Particle, best: Particle):
    """
    Barebones Particle Swarms is a variation of PSO proposed by Kennedy [1].

    [1] https://ieeexplore.ieee.org/abstract/document/1202251
    """
    mu = np.add(p.best, best.best) / 2
    sd = np.abs(np.subtract(p.best, best.best))
    p.pos = np.random.normal(mu, sd)
    fit = p.fitnessFunction(p.pos)
    if (fit < p.fitness):
        p.fitness = fit
        p.best = p.pos


def getChi(phi: float, k: float) -> float:
    return np.sqrt(k) if phi <= 4 else np.sqrt(
        (2*k) / (phi - 2 + np.sqrt(phi**2 - 4 * phi)))


class Particle(Individual):
    def __init__(
        self,
        fitnessFunction: FitnessFunction,
        bounds: Bounds,
        numDimensions: int
    ):
        super().__init__(
            fitnessFunction=fitnessFunction,
            bounds=bounds,
            numDimensions=numDimensions
        )
        self.prevPos = self.pos
