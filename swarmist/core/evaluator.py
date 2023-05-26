from typing import List
from dataclasses import dataclass
from functools import reduce
import numpy as np
from swarmist.core.dictionary import Pos, Bounds, Evaluation, FitnessFunction, ConstraintChecker

@dataclass(frozen=True)
class Evaluator:
    fit_func: FitnessFunction
    ndims: int
    bounds: Bounds
    constraints: List[ConstraintChecker]
    minimize: bool = True

    def clip(self, pos: Pos)->Pos:
        npos = np.clip(pos, self.bounds.min, self.bounds.max)
        if not self.constraints:
            return npos
        return reduce(lambda p, c: c(p), self.constraints, npos)
    
    def evaluate(self, pos: Pos)->Evaluation:
        npos = self.clip(pos)
        fit = self.fit_func(npos)
        return Evaluation(pos=npos, fit=fit)