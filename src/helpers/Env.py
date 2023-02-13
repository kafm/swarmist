from __future__ import annotations
from typing import TypeVar, Generic, List, Callable, Optional
import sys

T = TypeVar('T')

class Env:
    def __init__(self, 
        fitnessFunction: FitnessFunction,
        maxGenerations:Optional[int] = None,
        maxEvaluations:Optional[int] = None,
        minFitness: Optional[float] = None
    ):
        self.fitnessFunction = fitnessFunction
        self.evalEval = maxEvaluations != None
        self.evalFit = minFitness != None
        if maxGenerations == None:
            maxGenerations = sys.float_info.max if  self.evalEval or self.evalFit else 30
        self.maxGenerations = maxGenerations
        self.maxEvaluations = maxEvaluations
        self.minFitness = minFitness
        self.currEval:float = 0
        self.currFitness:float = sys.float_info.max
        self.currGen = 0
        
    def evaluate(self, pos: List[float])->float:
        if self.evalEval == True and self.currEval == self.maxEvaluations:
           raise MaxEvaluationReached(f"Max evals ({self.maxEvaluations}) stop condition reached")
        elif self.evalFit == True and self.minFitness <= self.currFitness:
           raise MinFitnessReached(f"Min fit ({self.minFitness}) stop condition reached")
        fit:float = self.fitnessFunction(pos)
        if fit < self.currFitness:
            self.currFitness = fit
        self.currEval += 1  
        return fit

    def next(self)->bool:
        self.currGen += 1
        return self.currGen < self.maxGenerations

class MaxEvaluationReached(Exception):
    pass

class MinFitnessReached(Exception):
    pass

class SearchResult(Generic[T]): 
    def __init__(self,  
        best: T,
        population: T, 
        fitnessByGeneration: List[float]
    ):
        self.best: T = best
        self.population: List[T] = population
        self.fitnessByGeneration:List[float] = fitnessByGeneration

class Bounds:
    def __init__(self, min:float=sys.float_info.min, max:float=sys.float_info.max):
        self.min:float = min
        self.max:float = max

    def __str__(self) -> str:
        return f"Min: {self.min}, Max: {self.max}."

FitnessFunction = Callable[[List[float]], float]