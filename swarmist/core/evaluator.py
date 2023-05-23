from typing import Callable
from dataclasses import dataclass
from functools import reduce
import numpy as np
from dictionary import Pos, Bounds, Evaluation, FitnessFunction

@dataclass(frozen=True)
class Evaluator:
    fit_func: FitnessFunction
    ndims: int
    bounds: Bounds
    minimize: bool = True
    constraints: Callable[[Pos], Pos] = []

    def clip(self, pos: Pos)->Pos:
        npos = np.clip(pos, self.bounds.min, self.bounds.max)
        return reduce(lambda p, c: c(p), self.constraints, npos)
    
    def evaluate(self, pos: Pos)->Evaluation:
        npos = self.clip(pos)
        fit = self.fit_func(npos)
        return Evaluation(pos=npos, fit=fit)