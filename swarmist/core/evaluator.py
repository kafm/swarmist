from typing import List, Optional
from dataclasses import dataclass
from functools import reduce
import numpy as np
from swarmist.core.dictionary import (
    Pos,
    Fit,
    Bounds,
    Evaluation,
    FitnessFunction,
    ConstraintChecker,
    SearchContext,
)


@dataclass(frozen=True)
class Evaluator:
    fit_func: FitnessFunction
    ndims: int
    bounds: Bounds
    constraints: List[ConstraintChecker]
    penalty_coefficient: bool
    minimize: bool = True

    def clip(self, pos: Pos) -> Pos:
        return np.clip(pos, self.bounds.min, self.bounds.max)

    def evaluate(self, pos: Pos, ctx: Optional[SearchContext] = None) -> Evaluation:
        npos = self.clip(pos)
        fit = self.fit_func(npos) + self.check_constraints(npos, ctx)
        return Evaluation(pos=npos, fit=fit)

    def check_constraints(self, pos: Pos, ctx: SearchContext) -> Fit:
        return (
            0
            if not self.constraints
            else sum([self._get_penalty(c(pos), ctx.curr_gen) for c in self.constraints])
        )

    def _get_penalty(self, value: Fit, k: int) -> Fit:
        return ( self.penalty_coefficient*k ) * abs(value)**2
