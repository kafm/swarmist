from __future__ import annotations
from typing import Optional
from pymonad.either import Right, Left, Either
import sys
import numpy as np
from .dictionary import *
from .errors import SearchEnded


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
        self.ndims = ndims
        self.bounds = bounds
        self.fit_func:FitnessFunction = fit_func
        self.constraints = constraints
        self.min_fit = min_fit
        self.max_evals = max_evals
        self.max_gen = sys.maxsize if not max_gen else max_gen
        self.curr_fit: Fit = sys.float_info.max
        self.curr_pos: Pos = None
        self.curr_gen:int = 0
        self.curr_eval:int = 0
        self._results:List[Evaluation] = [Evaluation()]
        self._ended = False

    def context(self)->SearchContext:
        return SearchContext(
             evaluate=self.evaluate, 
             clip=self.clip,
             ndims=self.ndims, 
             bounds=self.bounds,
             curr_gen=self.curr_gen,
             max_gen=self.max_gen,
             curr_fit=self.curr_fit,
             min_fit=self.min_fit,
             curr_eval =self.curr_eval,
             max_evals=self.max_evals
        )
    
    def clip(self, pos: Pos)->Pos:
        cs = self.constraints
        new_pos = np.clip(pos, self.bounds.min, self.bounds.max)
        return new_pos if not cs else cs(new_pos)

    def evaluate(self, pos :Pos)->Fit:
        self._assert_max_evals()
        self._assert_min_fit()
        fit = self.fit_func(pos)
        self.curr_eval += 1
        self._log_result(pos, fit)
        return fit 
    
    def run(self, callback: Callable)->Either[SearchResults, Exception]:
        while self._next():
            try: 
                callback(self.context())
            except Exception as e: 
                if not isinstance(e, SearchEnded):
                    return Left(e)
                self._log_result(self.curr_pos, self.curr_fit)
                break
        return Right(self._results)
        
    def _log_result(self, pos: Pos, fit: float):
        if fit <= self.curr_fit:
            self.curr_fit = fit
            self.curr_pos = pos
            self._results[self.curr_gen] = Evaluation(pos,fit)

    def _next(self)->bool:
        if not self._ended and self.curr_gen < self.max_gen:
            self.curr_gen += 1
            self._results.append(Evaluation(self.curr_pos, self.curr_fit))      
            return True
        return False  
    
    def _assert_max_evals(self):
        if self.max_evals and self.curr_eval >= self.max_evals:
            raise SearchEnded("Maximum number of evaluations reached")
        
    def _assert_min_fit(self):
        if self.min_fit and self.curr_fit <= self.min_fit:
            raise SearchEnded("Fitness goal reached")