from __future__ import annotations
from typing import Optional
from pymonad.either import Right, Either
from Dictionary import *
from Helpers import try_catch
from Exceptions import SearchEnded
import numpy as np
import sys

class SearchExecutor:
    def __init__(self, 
        fit_func: FitnessFunctionDef,
        ndims: int, 
        bounds: Bounds,
        constraints: Optional[ConstraintsChecker],
        max_gen: Optional[int],
        min_fit: Optional[float], 
        max_evals:  Optional[int],
    ):
        self.ctx = SearchContext(
             self.evaluate, 
             self.clip,
             ndims, 
             bounds
        )
        self.fit_func = fit_func.func
        self.constraints = constraints
        self.min_fit = min_fit
        self.max_evals = max_evals
        self.max_gen = max_gen
        self.curr_fit = sys.float_info.max
        self.curr_eval = 0
        self.curr_gen = 0

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
        self.curr_fit = min(fit, self.curr_fit)
        self.curr_eval += 1
        return fit 

    def next(self)->Either[int, Exception]:
        res = try_catch(self._assert_max_gen)
        if res.is_right:
            self.curr_gen += 1
            return Right(self.curr_gen)
        return res

    def results(self)->SearchResults:
        pass
    
    def _assert_max_evals(self):
        if self.max_evals and self.curr_eval >= self.max_evals:
            raise SearchEnded("Maximum number of evaluations reached")
        
    def _assert_min_fit(self):
        if self.min_fit and self.curr_fit <= self.min_fit:
            raise SearchEnded("Fitness goal reached")
        
    def _assert_max_gen(self):
        if self.max_gen and self.curr_gen >= self.max_gen:
            raise SearchEnded("Maximum number of generations reached")