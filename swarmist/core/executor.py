from __future__ import annotations
from typing import Optional
from pymonad.either import Right, Either
import sys
import numpy as np
from .dictionary import *
from .errors import SearchEnded, try_catch

class SearchExecutor:
    def __init__(self, 
        fit_func: FitnessFunctionDef,
        ndims: int, 
        bounds: Bounds,
        constraints: Optional[ConstraintsChecker],
        params: Optional[Parameters],
        max_gen: Optional[int],
        min_fit: Optional[float], 
        max_evals:  Optional[int],
    ):
        self.ctx = SearchContext(
             self.evaluate, 
             self.clip,
             ndims, 
             bounds,
             params
        )
        self.fit_func = fit_func.func
        self.constraints = constraints
        self.min_fit = min_fit
        self.max_evals = max_evals
        self.max_gen = max_gen
        self.curr_fit = sys.float_info.max
        self.curr_eval = 0
        self.curr_gen = -1
        self._evals_by_gen = []
        self.error = None

    def context(self)->SearchContext:
        return self.ctx
    
    def clip(self, pos: Pos)->Pos:
        lbound, ubound = self.ctx.bounds
        cs = self.constraints
        new_pos = np.clip(pos, lbound, ubound)
        return new_pos if not cs else cs(new_pos)

    def evaluate(self, pos :Pos)->Fit:
        self._assert_max_evals()
        self._assert_min_fit()
        fit = self.fit_func(pos)
        if fit < self.curr_fit:
            self.curr_fit = fit
            self._evals_by_gen[self.curr_gen] = Evaluation(pos, fit)
        self.curr_eval += 1
        return fit 

    def next(self)->Either[int, Exception]:
        res = try_catch(self._assert_max_gen)
        if res.is_right:
            self.curr_gen += 1
            self._evals_by_gen.append(None)
            return Right(self.curr_gen)
        else:
            self.error =  res.either(lambda e: e)
        return res

    def results(self)->List[Evaluation]:
        return self._evals_by_gen
    
    def _assert_max_evals(self):
        if self.max_evals and self.curr_eval >= self.max_evals:
            raise SearchEnded("Maximum number of evaluations reached")
        
    def _assert_min_fit(self):
        if self.min_fit and self.curr_fit <= self.min_fit:
            raise SearchEnded("Fitness goal reached")
        
    def _assert_max_gen(self):
        if self.max_gen and self.curr_gen >= self.max_gen:
            raise SearchEnded("Maximum number of generations reached")