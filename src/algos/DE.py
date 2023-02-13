from __future__ import annotations
from typing import Optional, List,  Tuple, cast, Any
import numpy as np

from helpers.Env import *
from helpers.Population import Neighborhood, Population, Individual, PopulationIterator

def DE(
    fitnessFunction: FitnessFunction,
    bounds: Bounds,
    numDimensions: int = 2,
    populationSize: int = 30,
    topology: Optional[str] = None,
    neighborhoodRange: Optional[int] = None,
    x: str = "rand",
    y: int = 1,
    z: str = "bin",  
    cr: float = .6,
    f: float = .5,
    gamma: float = .5,
    maxGenerations: Optional[int] = None,
    maxEvaluations: Optional[int] = None,
    minFitness: Optional[float] = None
) -> SearchResult[Agent]:
    """
    Differential Evolution implementation proposed by Storn [1].

    [1] https://ieeexplore.ieee.org/abstract/document/534789
    """
    env: Env = Env(
        fitnessFunction=fitnessFunction,
        maxGenerations=maxGenerations,
        maxEvaluations=maxEvaluations,
        minFitness=minFitness
    )
    agents: List[Agent] = [Agent(
        fitnessFunction=env.evaluate,
        bounds=bounds,
        numDimensions=numDimensions,
        index=i) for i in range(populationSize)]
    population: Population = Population(
        individuals=agents,
        topology=topology,
        neighborhoodRange=neighborhoodRange
    )
    return search(env, population, x, y, z, cr, f, f if not gamma else gamma)

def search(env: Env, population: Population, x: str, y: int, z: str, cr: float, f: float, gamma: float) -> SearchResult[Agent]:
    try:
        fitnessByGeneration: List[float] = []
        updateMethod = getUpdateMethod(x, y, z, cr, f, gamma)
        while (env.next()):
            iter: PopulationIterator[Agent] = population.iterator()
            while (iter.hasNext()):
                agent, neighbors = cast(Tuple[Agent, Neighborhood], iter.next())
                updateMethod(agent, neighbors)
            fitnessByGeneration.append(population.best().fitness)
    except MaxEvaluationReached:
        None
    except MinFitnessReached:
        None
    return SearchResult[Agent](
        best=population.best(),
        population=population.getAll(),
        fitnessByGeneration=fitnessByGeneration
    )

def  getUpdateMethod(x: str, y: int, z: str, cr: float, f: float, gamma: float)->UpdateMethod:
    crMethod: CrossoverMethod = getCrossoverMethod(z, cr)
    if x == "rand-to-best":
        return lambda a, n: randToBestUpdate(a, n, crMethod, y, f, gamma)
    elif x == "current-to-best":
        return lambda a, n: currentToBestUpdate(a, n, crMethod, y, f)
    elif x == "best":
       return lambda a, n: bestUpdate(a, n, crMethod, y, f)
    else:
       return lambda a, n: randUpdate(a, n, crMethod, y, f)

def randUpdate(agent: Agent, neighbors: Neighborhood, crMethod: CrossoverMethod, y: int, f: float):
    a: Agent = neighbors.getRandomIndividual(excludeIndexes=[agent.index])
    nY = y * 2
    choices: List[Agent] = neighbors.getRandomIndividuals(k=nY,excludeIndexes=[agent.index, a.index])
    diff = getYDiff(choices, nY, f)
    pos = crMethod(
        agent.pos,  
        lambda i : a.pos[i] + diff[i]
    )
    updatePosIfBetterFit(agent, pos)

def bestUpdate(agent: Agent, neighbors: Neighborhood, crMethod: CrossoverMethod, y: int, f: float):
    best: Agent = neighbors.best()
    nY = y * 2
    choices: List[Agent] = neighbors.getRandomIndividuals(k=nY,excludeIndexes=[agent.index, best.index])
    diff = getYDiff(choices, nY, f)
    pos = crMethod(
        agent.pos, 
        lambda i : best.pos[i] + diff[i]
    )
    updatePosIfBetterFit(agent, pos)

def randToBestUpdate(agent: Agent, neighbors: Neighborhood, crMethod: CrossoverMethod, y: int, f: float, gamma: float):
    best: Agent = neighbors.best()
    nY = y * 2
    a: Agent = neighbors.getRandomIndividual(excludeIndexes=[agent.index])
    choices: List[Agent] = neighbors.getRandomIndividuals(k=nY,excludeIndexes=[agent.index, best.index, a.index])
    a = gamma * best.pos + (1 - gamma) * a.pos  
    diff = getYDiff(choices, nY, f)
    pos = crMethod(
        agent.pos, 
        lambda i : a[i] + diff[i]
    )
    updatePosIfBetterFit(agent, pos)

def currentToBestUpdate(agent: Agent, neighbors: Neighborhood, crMethod: CrossoverMethod, y: int, f: float):
    best: Agent = neighbors.best()
    nY = y * 2
    choices: List[Agent] = neighbors.getRandomIndividuals(k=nY,excludeIndexes=[agent.index, best.index])
    a = agent.pos + f * (best.pos - agent.pos)
    diff = getYDiff(choices, nY , f)
    pos = crMethod(
        agent.pos, 
        lambda i : a[i] + diff[i]
    )
    updatePosIfBetterFit(agent, pos)

def getYDiff(agents: List[Agent], nY: int, f: float) -> List[float]:
    middlePoint = int(len(agents)/nY)
    return f * sum([
        agents[i].pos - agents[i+middlePoint].pos 
        for i in range(middlePoint)
    ]) 

def updatePosIfBetterFit(agent: Agent, pos: List[float]):
    fit = agent.fitnessFunction(pos)
    if fit < agent.fitness:
        agent.fitness = fit
        agent.pos = pos
        agent.best = pos

def getCrossoverMethod(crType: str, crProp: float)->CrossoverMethod:
    crMethod = exponentialCrossover if crType == "exp" else binomialCrossover
    return lambda pos, editMethod: crMethod(pos, editMethod, crProp)

def binomialCrossover(pos: List[float], editMethod: Callable[[int], float], cr: float)->List[float]:
    n = len(pos)
    r = np.random.randint(low=0, high = n, size = 1)
    newPos = np.copy(pos)
    for i in range(n):
        if np.random.uniform(0,1) < cr or i == r:
            newPos[i] = editMethod(i)
    return newPos

def exponentialCrossover(pos: List[float], editMethod: Callable[[int], float], cr: float)->List[float]:
    n = len(pos)
    j = n - 1
    r = np.random.randint(low=0, high = j, size = 1)
    newPos = np.copy(pos)
    for _ in range(j):
        if np.random.uniform(0,1) < cr:
            i = r + 1 if r < j else 0
            newPos[i] = editMethod(i) 
            r = i
        else:
            break
    return newPos

class Agent(Individual):
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

UpdateMethod = Callable[[Agent, Neighborhood], Any]
CrossoverMethod = Callable[[List[float], Callable[[int], float]], List[float]]