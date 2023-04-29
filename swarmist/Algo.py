from __future__ import annotations
from typing import List, Optional, Type, Callable
from dataclasses import dataclass, replace
import sys

def assertMinFitness(fit, min_fit):
    if fit < min_fit:
        raise Exception(f"Min fitness {min_fit} reached")

def assertMaxEvaluation(num_eval, max_evals):
    if num_eval > max_evals:
        raise Exception(f"Max evaluation {max_evals} reached")
    
def assertMaxGeneration(num_gen, max_gens):
    if num_gen > max_gens:
        raise Exception(f"Max generation {max_gens} reached")
    
@dataclass(frozen=True)
class StopCondition:
    min_fit: Optional[float]
    max_evals: Optional[int]
    max_gen: int = Optional[int]

FitnessFunction = Callable[[List[float]], float]

class WrappedFitness:
    def __init__(self, fitness_function: FitnessFunction, stop_condition: StopCondition):
        self.evals:int = 0
        self.min_fit: float = sys.float_info.max
        self.fitness_function: FitnessFunction = fitness_function
        self.max_evals = stop_condition.max_evals
        self.min_fit = stop_condition.min_fit

    def __call__(self, pos: List[float]) -> float:
        self.max_evals and assertMaxEvaluation(self.evals, self.max_evals)
        self.min_fit and assertMinFitness(self.min_fit, self.min_fit)
        fit = self.fitness_function(pos)
        self.min_fit = min(fit, self.min_fit)
        self.evals += 1
        return fit
    

def get_fitness_evaluator(fitness_function: FitnessFunction, stop_condition: StopCondition)->FitnessFunction:
    if not stop_condition.max_evals and not stop_condition.min_fit:
        return fitness_function     
    return WrappedFitness(fitness_function, stop_condition)
